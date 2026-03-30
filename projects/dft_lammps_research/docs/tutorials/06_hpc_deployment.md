# 06 - HPC集群使用指南 | HPC Deployment Guide

> **学习目标**: 掌握高性能计算集群的作业调度、资源管理和并行计算  
> **Learning Goal**: Master HPC job scheduling, resource management, and parallel computing

---

## 📋 目录 | Table of Contents

1. [HPC基础 | HPC Basics](#1-hpc基础--hpc-basics)
2. [作业调度系统 | Job Schedulers](#2-作业调度系统--job-schedulers)
3. [并行计算 | Parallel Computing](#3-并行计算--parallel-computing)
4. [工作流部署 | Workflow Deployment](#4-工作流部署--workflow-deployment)
5. [性能优化 | Performance Optimization](#5-性能优化--performance-optimization)
6. [故障排除 | Troubleshooting](#6-故障排除--troubleshooting)
7. [练习题 | Exercises](#7-练习题--exercises)

---

## 1. HPC基础 | HPC Basics

### 1.1 HPC架构 | HPC Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     HPC集群架构                                   │
│                   HPC Cluster Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                                               │
│   │  登录节点    │  Login Node                                   │
│   │  (交互使用)  │  (Interactive)                               │
│   └──────┬──────┘                                               │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────────────────────────────────────────┐         │
│   │              高速互联网络 (InfiniBand)             │         │
│   │           High-Speed Interconnect                 │         │
│   └──────────────────────────────────────────────────┘         │
│          │                                                      │
│     ┌────┴────┬────────┬────────┬────────┐                    │
│     ▼         ▼        ▼        ▼        ▼                    │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                │
│ │计算节点│ │计算节点│ │计算节点│ │计算节点│ │计算节点│                │
│ │GPU节点│ │GPU节点│ │CPU节点│ │CPU节点│ │大内存 │                │
│ │Node 1│ │Node 2│ │Node 3│ │Node 4│ │Node 5│                │
│ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                │
│    │        │        │        │        │                       │
│  4×GPU   4×GPU   64核    64核    1TB内存                      │
│                                                                 │
│ ┌────────────────────────────────────────────────────────────┐ │
│ │                    并行文件系统 (Lustre/GPFS)                │ │
│ │               Parallel File System                          │ │
│ └────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 常用命令 | Common Commands

```bash
# ========== 集群状态查看 | Check Cluster Status ==========

# 查看队列状态 (Slurm)
sinfo                    # 查看分区状态
sinfo -N -l             # 详细节点状态
squeue                  # 查看作业队列
squeue -u $USER         # 查看自己的作业

# 查看队列状态 (PBS)
qstat                   # 查看队列
qstat -u $USER          # 查看自己的作业
showq                   # 详细队列信息

# ========== 资源使用 | Resource Usage ==========

# 查看配额
lfs quota -u $USER /scratch    # Lustre文件系统配额

# 查看磁盘使用
du -sh ~
du -sh /scratch/$USER

# 查看作业详情 (Slurm)
scontrol show job JOBID
sacct -j JOBID --format=JobID,JobName,State,Elapsed,MaxRSS
```

---

## 2. 作业调度系统 | Job Schedulers

### 2.1 Slurm作业提交 | Slurm Job Submission

```bash
#!/bin/bash
#SBATCH -J vasp_job           # 作业名称 | Job name
#SBATCH -N 1                  # 节点数 | Number of nodes
#SBATCH --ntasks-per-node=64  # 每节点任务数 | Tasks per node
#SBATCH -t 24:00:00          # 运行时间 | Wall time (HH:MM:SS)
#SBATCH -p normal            # 分区 | Partition
#SBATCH --gres=gpu:4         # GPU资源 | GPU resources
#SBATCH -o %x-%j.out         # 标准输出 | STDOUT
#SBATCH -e %x-%j.err         # 标准错误 | STDERR
#SBATCH --mail-type=END,FAIL # 邮件通知 | Email notification
#SBATCH --mail-user=email@example.com

# 加载模块 | Load modules
module purge
module load vasp/6.3.0
module load intel/2021.4
module load impi/2021.4

# 设置环境变量 | Set environment
export OMP_NUM_THREADS=1
export I_MPI_FABRICS=shm:ofi

# 运行VASP | Run VASP
cd $SLURM_SUBMIT_DIR
mpirun -np $SLURM_NTASKS vasp_std
```

### 2.2 PBS作业提交 | PBS Job Submission

```bash
#!/bin/bash
#PBS -N vasp_job              # 作业名称
#PBS -l nodes=1:ppn=64        # 1节点, 64进程
#PBS -l walltime=24:00:00     # 运行时间
#PBS -q normal                # 队列
#PBS -j oe                    # 合并输出
#PBS -o job.out               # 输出文件
#PBS -M email@example.com     # 邮件地址
#PBS -m abe                   # 邮件通知时机

# 加载模块
module load vasp/6.3.0

# 进入工作目录
cd $PBS_O_WORKDIR

# 运行
mpirun -np 64 vasp_std
```

### 2.3 Python作业提交 | Python Job Submission

```python
#!/usr/bin/env python3
"""
HPC作业提交脚本 | HPC Job Submission Script
"""
import subprocess
import os
from pathlib import Path
from typing import List, Dict


class HPCScheduler:
    """HPC作业调度器"""
    
    def __init__(self, 
                 scheduler_type: str = "slurm",
                 default_partition: str = "normal"):
        """
        Args:
            scheduler_type: "slurm" 或 "pbs"
            default_partition: 默认分区
        """
        self.scheduler = scheduler_type
        self.partition = default_partition
    
    def submit_vasp_job(self,
                       work_dir: str,
                       ncores: int = 32,
                       walltime: str = "24:00:00",
                       vasp_cmd: str = "vasp_std") -> str:
        """
        提交VASP作业
        
        Returns:
            作业ID
        """
        work_dir = Path(work_dir)
        
        if self.scheduler == "slurm":
            script = f"""#!/bin/bash
#SBATCH -J vasp
#SBATCH -N 1
#SBATCH --ntasks-per-node={ncores}
#SBATCH -t {walltime}
#SBATCH -p {self.partition}
#SBATCH -o vasp-%j.out
#SBATCH -e vasp-%j.err

module load vasp

cd {work_dir}
mpirun -np {ncores} {vasp_cmd}
"""
            script_file = work_dir / "submit.sh"
            
        elif self.scheduler == "pbs":
            script = f"""#!/bin/bash
#PBS -N vasp
#PBS -l nodes=1:ppn={ncores}
#PBS -l walltime={walltime}
#PBS -q {self.partition}
#PBS -j oe

module load vasp

cd $PBS_O_WORKDIR
mpirun -np {ncores} {vasp_cmd}
"""
            script_file = work_dir / "submit.sh"
        
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
        
        # 写入脚本
        with open(script_file, 'w') as f:
            f.write(script)
        
        # 提交作业
        if self.scheduler == "slurm":
            result = subprocess.run(
                ['sbatch', str(script_file)],
                capture_output=True,
                text=True
            )
            job_id = result.stdout.strip().split()[-1]
        else:
            result = subprocess.run(
                ['qsub', str(script_file)],
                capture_output=True,
                text=True
            )
            job_id = result.stdout.strip()
        
        print(f"Submitted job {job_id}")
        return job_id
    
    def submit_python_job(self,
                         script_path: str,
                         work_dir: str,
                         ncores: int = 4,
                         walltime: str = "1:00:00",
                         conda_env: str = None) -> str:
        """提交Python作业"""
        
        work_dir = Path(work_dir)
        
        # 生成提交脚本
        script_content = f"""#!/bin/bash
#SBATCH -J python_job
#SBATCH -N 1
#SBATCH --ntasks-per-node={ncores}
#SBATCH -t {walltime}
#SBATCH -p {self.partition}

"""
        if conda_env:
            script_content += f"""
source ~/anaconda3/etc/profile.d/conda.sh
conda activate {conda_env}
"""
        
        script_content += f"""
cd {work_dir}
python {script_path}
"""
        
        script_file = work_dir / "submit_python.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # 提交
        result = subprocess.run(
            ['sbatch', str(script_file)],
            capture_output=True,
            text=True
        )
        
        job_id = result.stdout.strip().split()[-1]
        return job_id
    
    def check_job_status(self, job_id: str) -> str:
        """检查作业状态"""
        if self.scheduler == "slurm":
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h', '-o', '%T'],
                capture_output=True,
                text=True
            )
            return result.stdout.strip() or "COMPLETED"
        else:
            result = subprocess.run(
                ['qstat', '-f', job_id],
                capture_output=True,
                text=True
            )
            # 解析状态
            return "UNKNOWN"
    
    def cancel_job(self, job_id: str):
        """取消作业"""
        if self.scheduler == "slurm":
            subprocess.run(['scancel', job_id])
        else:
            subprocess.run(['qdel', job_id])


# 使用示例
if __name__ == "__main__":
    scheduler = HPCScheduler(scheduler_type="slurm")
    
    # 提交VASP作业
    job_id = scheduler.submit_vasp_job(
        work_dir="./calculation",
        ncores=32,
        walltime="12:00:00"
    )
    
    print(f"Job ID: {job_id}")
```

---

## 3. 并行计算 | Parallel Computing

### 3.1 MPI并行 | MPI Parallelization

```python
"""
MPI并行计算示例 | MPI Parallel Computing Example
"""
from mpi4py import MPI
import numpy as np


def mpi_parallel_demo():
    """MPI并行演示"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 每个进程处理一部分数据
    data_size = 1000
    local_size = data_size // size
    
    if rank == 0:
        # 主进程生成数据
        full_data = np.random.rand(data_size)
    else:
        full_data = None
    
    # 分发数据
    local_data = np.zeros(local_size)
    comm.Scatter(full_data, local_data, root=0)
    
    # 本地计算
    local_result = np.sum(local_data ** 2)
    
    # 收集结果
    total_result = comm.reduce(local_result, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"Total result: {total_result}")


# VASP MPI配置优化
def vasp_mpi_optimization():
    """
    VASP MPI优化建议
    
    1. 纯MPI模式
       export OMP_NUM_THREADS=1
       mpirun -np 64 vasp_std
    
    2. MPI+OpenMP混合模式 (适合大体系)
       export OMP_NUM_THREADS=4
       export OMP_PLACES=cores
       mpirun -np 16 vasp_std
    
    3. k点并行 (NCORE)
       NCORE = 4  # 每k点使用4核
       # 64核系统可同时处理16个k点
    """
    pass
```

### 3.2 数组作业 | Array Jobs

```bash
#!/bin/bash
#SBATCH -J array_job
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -t 2:00:00
#SBATCH --array=1-100%10   # 100个任务, 同时运行10个

# 获取数组索引
INDEX=$SLURM_ARRAY_TASK_ID

# 根据索引选择输入
cd ./job_$INDEX
mpirun -np 16 vasp_std
```

```python
# Python数组作业管理
class ArrayJobManager:
    """数组作业管理器"""
    
    def submit_array_job(self, n_tasks: int, max_parallel: int = 10):
        """提交数组作业"""
        
        script = f"""#!/bin/bash
#SBATCH -J array
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -t 2:00:00
#SBATCH --array=1-{n_tasks}%{max_parallel}

INDEX=$SLURM_ARRAY_TASK_ID
cd ./task_$INDEX
python run_task.py --index $INDEX
"""
        with open('array_submit.sh', 'w') as f:
            f.write(script)
        
        subprocess.run(['sbatch', 'array_submit.sh'])
```

---

## 4. 工作流部署 | Workflow Deployment

### 4.1 批量作业提交 | Batch Job Submission

```python
"""
批量HPC作业提交
Batch HPC Job Submission
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor


class BatchHPCManager:
    """批量HPC作业管理器"""
    
    def __init__(self, scheduler: HPCScheduler, max_jobs: int = 10):
        """
        Args:
            scheduler: HPC调度器
            max_jobs: 最大同时运行作业数
        """
        self.scheduler = scheduler
        self.max_jobs = max_jobs
        self.submitted_jobs = {}
    
    def submit_batch(self, work_dirs: List[str], **kwargs) -> Dict[str, str]:
        """
        批量提交作业
        
        Args:
            work_dirs: 工作目录列表
            **kwargs: 传递给submit_vasp_job的参数
            
        Returns:
            {work_dir: job_id}
        """
        job_map = {}
        
        for i, work_dir in enumerate(work_dirs):
            # 等待有可用槽位
            while len(self.submitted_jobs) >= self.max_jobs:
                self._update_job_status()
                time.sleep(10)
            
            # 提交作业
            job_id = self.scheduler.submit_vasp_job(work_dir, **kwargs)
            job_map[work_dir] = job_id
            self.submitted_jobs[job_id] = {'status': 'PENDING', 'work_dir': work_dir}
            
            print(f"[{i+1}/{len(work_dirs)}] Submitted {job_id} for {work_dir}")
        
        return job_map
    
    def _update_job_status(self):
        """更新作业状态"""
        for job_id in list(self.submitted_jobs.keys()):
            status = self.scheduler.check_job_status(job_id)
            self.submitted_jobs[job_id]['status'] = status
            
            # 移除已完成作业
            if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                del self.submitted_jobs[job_id]
    
    def wait_all(self, poll_interval: int = 60):
        """等待所有作业完成"""
        while self.submitted_jobs:
            self._update_job_status()
            
            running = sum(1 for j in self.submitted_jobs.values() if j['status'] == 'RUNNING')
            pending = sum(1 for j in self.submitted_jobs.values() if j['status'] == 'PENDING')
            
            print(f"Running: {running}, Pending: {pending}, Total: {len(self.submitted_jobs)}")
            
            if self.submitted_jobs:
                time.sleep(poll_interval)
```

### 4.2 工作流依赖 | Workflow Dependencies

```bash
#!/bin/bash
# 作业依赖示例 | Job Dependency Example

# 提交第1个作业
JOB1=$(sbatch --parsable job1.sh)
echo "Submitted job1: $JOB1"

# 提交第2个作业，依赖job1完成
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 job2.sh)
echo "Submitted job2: $JOB2"

# 提交第3个作业，依赖job2完成
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 job3.sh)
echo "Submitted job3: $JOB3"

# 或者并行后汇总
# JOB4=$(sbatch --parsable --dependency=afterok:$JOB1:$JOB2:$JOB3 job4.sh)
```

---

## 5. 性能优化 | Performance Optimization

### 5.1 VASP性能调优 | VASP Performance Tuning

```bash
# ========== VASP性能优化参数 | VASP Performance Parameters ==========

# 1. k点并行 (NCORE)
# NCORE决定每个k点/能带组使用多少核心
# 推荐: NCORE = sqrt(总核心数)
NCORE = 8          # 64核系统推荐

# 2. 节点内通信 (KPAR)
KPAR = 4           # k点间并行
# 总核心数 = KPAR × NCORE × NPAR (通常NPAR=1)

# 3. 内存优化
LSCALU = .TRUE.    # 大体系使用ScaLAPACK
NSIM = 4           # 同时优化的能带数

# 4. FFT优化
# 选择适合体系大小的FFT网格
NGX, NGY, NGZ      # 应满足 2^n × 3^m × 5^l
```

### 5.2 性能监控 | Performance Monitoring

```python
"""
作业性能监控 | Job Performance Monitoring
"""
import subprocess
import json


def monitor_job_performance(job_id: str):
    """监控作业性能"""
    
    # 使用sacct获取资源使用
    result = subprocess.run(
        ['sacct', '-j', job_id, '--format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize,NNodes,NCPUS,TotalCPU', '-P'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    # 计算效率
    # CPU效率 = TotalCPU / (Elapsed × NCPUS)
    # 内存效率 = MaxRSS / 可用内存


def optimize_resource_allocation(history_file: str):
    """根据历史作业优化资源分配"""
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # 分析资源使用模式
    # 推荐合适的核心数、内存、运行时间
    
    recommendations = {
        'typical_cores': 32,
        'typical_memory_gb': 64,
        'typical_walltime': '12:00:00'
    }
    
    return recommendations
```

---

## 6. 故障排除 | Troubleshooting

### 6.1 常见错误 | Common Errors

| 错误 | Error | 原因 | Cause | 解决方案 | Solution |
|------|-------|------|-------|---------|----------|
| `OOM` | 内存不足 | 请求内存不够 | Insufficient memory | 增加--mem参数 |
| `TIMEOUT` | 超时 | 运行时间设置太短 | Wall time too short | 增加-t参数 |
| `NODE_FAIL` | 节点故障 | 计算节点问题 | Node hardware issue | 重新提交 |
| `CANCELLED` | 被取消 | 超过配额或手动取消 | Quota exceeded | 检查配额 |
| `FAILED` | 失败 | 程序错误 | Program error | 查看错误日志 |

### 6.2 调试脚本 | Debugging Script

```bash
#!/bin/bash
# 调试模式作业脚本 | Debug Mode Job Script

#SBATCH -J debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:30:00
#SBATCH -p debug          # 使用调试分区
#SBATCH --gres=gpu:1      # 测试GPU

# 环境检查
echo "=== Environment Check ==="
echo "Hostname: $(hostname)"
echo "PWD: $(pwd)"
echo "Modules:"
module list
echo ""
echo "MPI Info:"
mpirun --version
echo ""
echo "GPU Info:"
nvidia-smi
echo ""

# 测试MPI
echo "=== MPI Test ==="
mpirun -np 4 hostname

# 运行实际程序
echo "=== Running Program ==="
mpirun -np 4 ./your_program
```

---

## 7. 练习题 | Exercises

### 练习 1: 数组作业管理

```python
# 实现数组作业的自动提交和监控
# 用于高通量筛选

def submit_ht_screening(n_structures: int):
    """提交高通量筛选数组作业"""
    # 为每个结构生成输入
    # 提交数组作业
    # 监控完成状态
    pass
```

### 练习 2: 资源优化

```python
# 分析历史作业，自动优化资源请求
# 根据体系大小推荐合适的核心数和时间

def recommend_resources(structure_file: str) -> dict:
    """根据结构推荐资源"""
    # 读取结构，计算原子数
    # 查询历史相似计算
    # 返回推荐配置
    pass
```

---

**下一步**: [07 - 高级工作流定制](07_advanced_workflows.md)
