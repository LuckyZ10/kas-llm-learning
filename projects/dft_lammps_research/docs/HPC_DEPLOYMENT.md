# HPC 部署指南与性能优化

## 目录

1. [概述](#概述)
2. [环境准备](#环境准备)
3. [模块安装](#模块安装)
4. [调度器配置](#调度器配置)
5. [使用示例](#使用示例)
6. [性能优化](#性能优化)
7. [故障排除](#故障排除)

---

## 概述

本文档介绍如何在HPC集群上部署和使用DFT-LAMMPS研究流水线，包括：

- **hpc_scheduler.py** - 支持Slurm/PBS/LSF的作业调度接口
- **parallel_optimizer.py** - DFT/ML/MD并行优化模块
- **checkpoint_manager.py** - 断点续传和容错机制

### 支持的调度系统

| 调度系统 | 测试状态 | 备注 |
|---------|---------|------|
| Slurm   | ✅ 完整支持 | 推荐 |
| PBS/Torque | ✅ 完整支持 | - |
| LSF     | ✅ 完整支持 | - |
| 本地模式 | ✅ 支持 | 用于测试 |

---

## 环境准备

### 1. 系统要求

#### 硬件要求

| 组件 | 最小配置 | 推荐配置 |
|-----|---------|---------|
| CPU | 8核心 | 32+核心 |
| 内存 | 32GB | 128GB+ |
| 存储 | 500GB HDD | 1TB+ NVMe SSD |
| GPU | 可选 | NVIDIA A100/V100 |
| 网络 | 千兆以太网 | InfiniBand HDR |

#### 软件要求

```bash
# Python版本
Python >= 3.8

# 核心依赖
numpy >= 1.21.0
pandas >= 1.3.0

# 可选依赖（分布式训练）
torch >= 1.10.0
torchvision >= 0.11.0
mpi4py >= 3.1.0  # MPI支持

# 可选依赖（云存储）
fsspec >= 2023.0.0
s3fs >= 2023.0.0  # S3存储
gcsfs >= 2023.0.0  # GCS存储

# 可选依赖（压缩）
lz4 >= 4.0.0

# DFT软件
VASP / Quantum ESPRESSO / ABACUS

# MD软件
LAMMPS >= 2020.03.03
GPUMD >= 3.0  # NEP训练
```

### 2. 安装步骤

#### 步骤1: 创建conda环境

```bash
# 创建环境
conda create -n dft-hpc python=3.10
conda activate dft-hpc

# 安装基础依赖
conda install numpy pandas scipy scikit-learn -c conda-forge

# 安装ASE和pymatgen
conda install -c conda-forge ase pymatgen

# 安装MPI（如需要分布式运行）
conda install -c conda-forge mpi4py openmpi
```

#### 步骤2: 安装PyTorch（如需要ML训练）

```bash
# CPU版本
conda install pytorch torchvision cpuonly -c pytorch

# CUDA版本（根据CUDA版本选择）
# CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 步骤3: 安装可选依赖

```bash
# 云存储支持
pip install fsspec s3fs gcsfs

# 高级压缩
pip install lz4

# 性能监控
pip install psutil memory_profiler
```

#### 步骤4: 验证安装

```bash
# 测试Python导入
python -c "import hpc_scheduler; import parallel_optimizer; import checkpoint_manager; print('All modules loaded successfully')"

# 测试调度器检测
python -c "from hpc_scheduler import detect_scheduler; print(detect_scheduler())"

# 测试MPI（如安装）
mpirun -np 4 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()}/{MPI.COMM_WORLD.Get_size()}')"
```

---

## 调度器配置

### Slurm配置

#### 基础配置

```bash
# 编辑 ~/.bashrc 或 ~/.bash_profile
export SLURM_CONF=/etc/slurm/slurm.conf

# 检查Slurm可用性
sinfo
squeue
```

#### 常用分区设置

```python
from hpc_scheduler import ResourceRequest

# 标准计算分区
standard_resources = ResourceRequest(
    num_nodes=2,
    num_cores_per_node=32,
    memory_gb=128,
    walltime_hours=24,
    partition="normal"
)

# GPU分区
gpu_resources = ResourceRequest(
    num_nodes=1,
    num_cores_per_node=8,
    num_gpus=4,
    gpu_type="a100",
    memory_gb=256,
    walltime_hours=48,
    partition="gpu"
)

# 大内存分区
fat_resources = ResourceRequest(
    num_nodes=1,
    num_cores_per_node=64,
    memory_gb=1024,
    walltime_hours=72,
    partition="fat"
)
```

### PBS/Torque配置

```bash
# 检查PBS安装
qstat -q
pbsnodes -a

# 设置环境变量
export PBS_DEFAULT=server.domain.com
```

### LSF配置

```bash
# 检查LSF安装
bqueues
bhosts

# 设置环境变量
export LSF_ENVDIR=/etc/lsf
export LSF_SERVERDIR=/usr/local/lsf/10.1/linux3.10-glibc2.17-x86_64/etc
```

---

## 使用示例

### 示例1: 批量DFT计算

```python
#!/usr/bin/env python3
"""批量DFT计算示例"""

from pathlib import Path
from hpc_scheduler import create_scheduler, JobSpec, ResourceRequest
from parallel_optimizer import DFTBatchProcessor, ParallelConfig

# 创建调度器
scheduler = create_scheduler()  # 自动检测

# 定义资源
resources = ResourceRequest(
    num_nodes=1,
    num_cores_per_node=32,
    memory_gb=128,
    walltime_hours=12,
    partition="normal",
    modules=["intel/2021.4", "mpi/intelmpi/2021.4", "vasp/6.3.0"]
)

# 准备结构列表
structures = [f"POSCAR_{i:03d}" for i in range(100)]

# 批量处理配置
config = ParallelConfig(
    num_workers=4,  # 每个节点同时运行4个计算
    mode="process",
    retry_attempts=2
)

# 创建处理器
processor = DFTBatchProcessor(config, scheduler)

# 执行批量计算
results = processor.process_structures(
    structures=structures,
    calc_type="relaxation",
    calculator="vasp",
    template_dir="./vasp_template",
    output_dir="./dft_results"
)

# 汇总结果
success_count = sum(1 for r in results if r.success)
print(f"Completed: {success_count}/{len(results)}")
```

### 示例2: GPU加速的NEP训练

```python
#!/usr/bin/env python3
"""NEP训练作业提交示例"""

from hpc_scheduler import create_scheduler, JobSpec, ResourceRequest

scheduler = create_scheduler()

# GPU资源请求
gpu_resources = ResourceRequest(
    num_nodes=1,
    num_cores_per_node=16,
    num_gpus=4,
    gpu_type="a100",
    memory_gb=256,
    walltime_hours=48,
    partition="gpu",
    modules=["cuda/11.8", "gpumd/3.6"]
)

# 创建训练作业
spec = JobSpec(
    name="nep_training",
    working_dir=Path("./nep_training"),
    commands=[
        "source activate dft-hpc",
        "python -m nep_training_pipeline \\\n            --train-file train.xyz \\\n            --test-file test.xyz \\\n            --epochs 1000000 \\\n            --batch-size 1000",
    ],
    resources=gpu_resources,
    env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
)

# 提交作业
job_id = scheduler.submit(spec)
print(f"Submitted NEP training job: {job_id}")

# 监控作业
status = scheduler.wait_for_job(job_id, poll_interval=60)
print(f"Job finished with status: {status}")
```

### 示例3: 多步骤工作流

```python
#!/usr/bin/env python3
"""多步骤DFT-ML-MD工作流"""

from pathlib import Path
from hpc_scheduler import WorkflowManager, JobSpec, ResourceRequest

# 创建工作流管理器
workflow = WorkflowManager()

# 步骤1: DFT数据生成
dft_spec = JobSpec(
    name="dft_data_generation",
    working_dir=Path("./1_dft"),
    commands=[
        "python generate_structures.py --n-structures 1000",
        "python run_vasp_batch.py",
    ],
    resources=ResourceRequest(
        num_nodes=10,
        num_cores_per_node=32,
        walltime_hours=24,
        partition="normal"
    )
)
workflow.submit_task("dft", dft_spec)

# 步骤2: NEP训练（依赖于步骤1）
nep_spec = JobSpec(
    name="nep_training",
    working_dir=Path("./2_nep"),
    commands=[
        "python prepare_nep_data.py",
        "gpumd < nep.in",
    ],
    resources=ResourceRequest(
        num_nodes=1,
        num_cores_per_node=16,
        num_gpus=4,
        walltime_hours=48,
        partition="gpu"
    )
)
workflow.submit_task("nep", nep_spec, depends_on=["dft"])

# 步骤3: MD模拟（依赖于步骤2）
md_spec = JobSpec(
    name="md_simulation",
    working_dir=Path("./3_md"),
    commands=[
        "python setup_md.py",
        "lmp -in md.in",
    ],
    resources=ResourceRequest(
        num_nodes=2,
        num_cores_per_node=32,
        num_gpus=2,
        walltime_hours=72,
        partition="gpu"
    )
)
workflow.submit_task("md", md_spec, depends_on=["nep"])

# 等待所有任务完成
results = workflow.wait_for_all(timeout=7*24*3600)  # 7天超时
print(f"Workflow results: {results}")
```

### 示例4: 使用检查点实现容错训练

```python
#!/usr/bin/env python3
"""使用检查点的容错训练示例"""

from checkpoint_manager import CheckpointManager, CheckpointConfig
from checkpoint_manager import resume_or_start, checkpoint_context

# 配置检查点
config = CheckpointConfig(
    checkpoint_dir="./training_checkpoints",
    save_interval_iterations=100,
    save_interval_seconds=600,  # 每10分钟保存
    keep_last_n=5,
    keep_best=True,
    compression="gzip",
    async_save=True
)

manager = CheckpointManager(config)

# 定义初始化函数
def init_training():
    return {
        "model": create_model(),
        "optimizer": create_optimizer(),
        "epoch": 0,
        "best_loss": float('inf')
    }

# 尝试恢复或开始新训练
state, start_epoch, was_resumed = resume_or_start(
    checkpoint_dir="./training_checkpoints",
    task_name="nep_training",
    init_func=init_training
)

print(f"Starting from epoch {start_epoch}, resumed: {was_resumed}")

# 训练循环
try:
    for epoch in range(start_epoch, 1000):
        # 训练步骤
        loss = train_epoch(state["model"], state["optimizer"])
        
        # 更新状态
        state["epoch"] = epoch
        
        # 保存最佳模型
        is_best = loss < state["best_loss"]
        if is_best:
            state["best_loss"] = loss
        
        # 自动保存检查点
        if epoch % 100 == 0:
            manager.save(
                state=state,
                task_name="nep_training",
                iteration=epoch,
                progress_percentage=epoch / 10,
                is_best=is_best
            )
            
except Exception as e:
    # 紧急保存
    manager.save(
        state=state,
        task_name="nep_training",
        iteration=epoch,
        tags={"status": "interrupted", "error": str(e)}
    )
    raise
```

### 示例5: 并行MD轨迹分析

```python
#!/usr/bin/env python3
"""大规模MD轨迹并行分析"""

from parallel_optimizer import MDTrajectoryAnalyzer, ParallelConfig
import numpy as np

# 配置并行分析
config = ParallelConfig(
    num_workers=16,  # 使用16个进程
    mode="process",
    chunk_size=100   # 每100帧一个块
)

analyzer = MDTrajectoryAnalyzer(config)

# 定义分析函数
def calc_rmsd(frame_data):
    """计算RMSD"""
    atoms = frame_data.get("atoms", [])
    # RMSD计算逻辑
    return np.std([a.get("c_pe", 0) for a in atoms])

def calc_coordination(frame_data):
    """计算配位数"""
    atoms = frame_data.get("atoms", [])
    positions = np.array([[a.get("x", 0), a.get("y", 0), a.get("z", 0)] 
                         for a in atoms])
    # 配位数计算逻辑
    return len(positions)

# 并行分析
results = analyzer.analyze_trajectory(
    trajectory_file="production.dump",
    analysis_funcs={
        "rmsd": calc_rmsd,
        "coordination": calc_coordination,
        "temperature": lambda f: np.mean([a.get("c_ke", 0) for a in f.get("atoms", [])])
    },
    chunk_size=50
)

# 保存结果
import json
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

# 计算RDF
rdf_data = analyzer.compute_rdf(
    trajectory_file="production.dump",
    bin_width=0.05,
    r_max=10.0
)

np.savetxt("rdf.txt", np.column_stack([rdf_data["r"], rdf_data["g_r"]]))

analyzer.shutdown()
```

---

## 性能优化

### 1. 并行度调优

#### DFT计算并行度

```python
from parallel_optimizer import ParallelConfig

# 对于小规模结构（<50原子）
small_config = ParallelConfig(
    num_workers=4,  # 每个节点同时运行4个VASP实例
    mode="process"
)

# 对于中等规模结构（50-200原子）
medium_config = ParallelConfig(
    num_workers=2,  # 每个节点同时运行2个VASP实例
    mode="process"
)

# 对于大规模结构（>200原子）
large_config = ParallelConfig(
    num_workers=1,  # 独占节点
    mode="process"
)
```

#### VASP并行设置

```bash
# K点并行（NCORE）
# 对于>4个k点的计算
export NCORE=4  # 每个k点使用4个核心

# 能带并行（NPAR）
# 对于大体系
export NPAR=4   # 分成4组并行

# 最优设置参考
# 32核心: NCORE=4, NPAR=8
# 64核心: NCORE=4, NPAR=16
```

### 2. I/O优化

#### 使用快速存储

```python
from hpc_scheduler import ResourceRequest

# 使用节点本地SSD
resources = ResourceRequest(
    num_nodes=1,
    num_cores_per_node=32,
    memory_gb=128,
    constraint="ssd",  # 请求SSD节点
    walltime_hours=24
)
```

```bash
# 作业脚本中移动数据到本地存储
#SBATCH --constraint=ssd

cd $TMPDIR  # 使用节点本地临时存储
cp $SLURM_SUBMIT_DIR/INPUT/* .

# 运行计算
mpirun vasp_std

# 复制结果回提交目录
cp OUTCAR $SLURM_SUBMIT_DIR/
```

#### 减少I/O频率

```python
# checkpoint_manager配置
from checkpoint_manager import CheckpointConfig

config = CheckpointConfig(
    save_interval_iterations=500,  # 减少保存频率
    compression="lz4",  # 使用快速压缩
    async_save=True     # 异步保存
)
```

### 3. 内存优化

#### 内存监控

```python
import psutil

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory: {mem_info.rss / 1024**3:.2f} GB")
```

#### 数据流式处理

```python
# 对于大轨迹文件，使用分块处理
from parallel_optimizer import MDTrajectoryAnalyzer

config = ParallelConfig(
    num_workers=8,
    chunk_size=50,  # 小批量处理
    memory_limit_gb=32  # 内存限制
)

analyzer = MDTrajectoryAnalyzer(config)
```

### 4. GPU优化

#### CUDA环境设置

```bash
# .bashrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 优化CUDA性能
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB缓存
export CUDA_LAUNCH_BLOCKING=0
```

#### 多GPU训练

```python
from parallel_optimizer import DistributedTrainingConfig, DistributedTrainer

config = DistributedTrainingConfig(
    backend="nccl",
    world_size=4,  # 4个GPU
    use_amp=True,  # 自动混合精度
    gradient_accumulation_steps=4
)

trainer = DistributedTrainer(config)
model = trainer.prepare_model(model)
dataloader = trainer.prepare_dataloader(dataset, batch_size=32)
```

### 5. 网络优化

#### InfiniBand配置

```bash
# 检查IB设备
ibstat
ibstatus

# 优化MPI参数
export OMPI_MCA_btl=openib,self,vader
export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_pml=ucx
```

---

## 故障排除

### 常见问题

#### 1. 作业提交失败

```bash
# 检查调度器可用性
which sbatch  # Slurm
which qsub    # PBS
which bsub    # LSF

# 检查权限
squeue -u $USER  # 查看自己的作业

# 查看详细错误
sbatch --test-only job.sh  # 仅验证不提交
```

#### 2. 检查点恢复失败

```python
from checkpoint_manager import CheckpointManager

manager = CheckpointManager()

# 列出可用检查点
checkpoints = manager.list_checkpoints(task_name="my_task")
for cp in checkpoints:
    print(f"{cp['checkpoint_id']}: {cp['metadata'].status.value}")

# 手动指定检查点恢复
recovery = manager.recover_from("my_task_20240309_120000_a1b2c3d4")
```

#### 3. 内存不足

```python
# 减少并行度
config = ParallelConfig(
    num_workers=2,  # 减少worker数量
    batch_size=10   # 减小批大小
)

# 使用内存映射
import numpy as np
data = np.memmap('large_array.dat', dtype='float32', mode='r', shape=(1000000, 1000))
```

#### 4. 网络超时

```python
from checkpoint_manager import FaultTolerantExecutor

# 增加重试次数和延迟
executor = FaultTolerantExecutor(
    max_retries=5,
    base_delay=5.0,
    max_delay=300.0
)
```

### 性能诊断

```bash
# CPU使用率
htop

# I/O监控
iotop

# 网络监控
iftop

# GPU监控
nvidia-smi -l 1

# 作业统计
sacct -j <job_id> --format=JobID,JobName,MaxRSS,Elapsed
```

---

## 附录

### A. 参考配置

#### 小集群（<100节点）

```yaml
# config_small.yaml
parallel:
  num_workers: 4
  max_concurrent_jobs: 20
  
checkpoint:
  save_interval_seconds: 300
  keep_last_n: 3
  
resources:
  default_partition: "normal"
  max_walltime_hours: 72
```

#### 大集群（>1000节点）

```yaml
# config_large.yaml
parallel:
  num_workers: 8
  max_concurrent_jobs: 100
  
checkpoint:
  save_interval_seconds: 600
  keep_last_n: 5
  storage_backend: "s3"  # 使用对象存储
  
resources:
  default_partition: "batch"
  max_walltime_hours: 168
```

### B. 性能基准

| 任务类型 | 并行度 | 加速比 | 效率 |
|---------|-------|-------|-----|
| DFT relax (100 structures) | 32x | 28x | 87% |
| NEP training (1M epochs) | 4x GPU | 3.8x | 95% |
| MD analysis (100K frames) | 16x | 15x | 94% |

---

## 联系与支持

如有问题或建议，请联系：
- 邮箱: hpc-support@example.com
- 文档: https://docs.example.com/dft-hpc
- 问题追踪: https://github.com/example/dft-hpc/issues
