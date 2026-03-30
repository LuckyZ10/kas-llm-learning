# DFT计算的GPU加速指南

本文档详细介绍如何在DFT计算中利用GPU加速，包括VASP和Quantum ESPRESSO的GPU版本配置与优化。

---

## 1. GPU加速原理

### 1.1 为什么GPU适合DFT

**DFT中的可并行操作**:
| 操作 | 计算占比 | GPU加速比 | 实现难度 |
|------|----------|-----------|----------|
| FFT (3D) | 30-50% | 5-10x | 低 |
| 矩阵乘法 | 20-30% | 10-20x | 低 |
| 对角化 | 15-25% | 3-5x | 中 |
| 非局部投影 | 10-15% | 2-4x | 中 |

**GPU架构优势**:
- **高内存带宽**: HBM2e/HBM3 可达 2-3 TB/s (CPU: ~100 GB/s)
- **大规模并行**: 数千个CUDA核心同时工作
- **适合规则计算**: FFT和矩阵乘法的完美匹配

### 1.2 硬件选择

**NVIDIA GPU对比 (2024-2025)**:
| GPU | 显存 | CUDA核心 | 显存带宽 | 适用场景 |
|-----|------|----------|----------|----------|
| A100 | 40/80GB | 6912 | 2.0 TB/s | 大规模计算 |
| H100 | 80GB | 16896 | 3.4 TB/s | AI+科学计算 |
| RTX 4090 | 24GB | 16384 | 1.0 TB/s | 小型工作站 |
| L40S | 48GB | 18176 | 0.9 TB/s | 推理+轻量训练 |

**建议配置**:
```
小体系 (<100原子): 1×A100 40GB
中体系 (100-500原子): 2-4×A100 80GB
大体系 (>500原子): 4-8×H100 80GB
```

---

## 2. VASP GPU加速

### 2.1 版本要求

**支持的VASP版本**:
- VASP 6.3.0+: 初步GPU支持
- VASP 6.4.0+: 稳定GPU支持
- VASP 6.5.0+: 增强BSE/分子动力学GPU支持

**编译要求**:
```bash
# 需要CUDA Toolkit 11.0+
# 需要NVIDIA HPC SDK或PGI编译器
# FFTW3或cuFFT
```

### 2.2 编译配置

**Makefile.include (GPU版本)**:
```makefile
# VASP 6.5 GPU编译配置

# 预处理选项
CPP_OPTIONS = -DHOST=\"LinuxNV_GPU\" \
              -DMPI -DMPI_BLOCK=8000 \
              -Duse_collective \
              -DscaLAPACK \
              -DCACHE_SIZE=4000 \
              -Davoidalloc \
              -Dvasp6 \
              -Duse_bse_te \
              -Dtbdyn \
              -Duse_shmem \
              -DVASP2WANNIER90v2

CPP         = nvcc -E -P -C $*$(FUFFIX) >$*$(SUFFIX) $(CPP_OPTIONS)

FC          = mpif90 -acc -gpu=cc80,cuda11.0
FCL         = mpif90 -acc -gpu=cc80,cuda11.0 -c++libs

FREE        = -Mfree

FFLAGS      = -Mbackslash -Mlarge_arrays

OFLAG       = -fast
OFLAG_IN    = $(OFLAG)
DEBUG       = -fast -g -C

# CUDA
CUDA        = -Mcuda=cuda11.0
CUDAFLAGS   = -acc -gpu=cc80,cuda11.0

# BLAS/LAPACK/ScaLAPACK
MKL_PATH    = /opt/intel/oneapi/mkl/latest/lib/intel64
BLAS        = -L$(MKL_PATH) -lmkl_blas95_lp64
LAPACK      = -L$(MKL_PATH) -lmkl_lapack95_lp64
BLACS       = -L$(MKL_PATH) -lmkl_blacs_openmpi_lp64
SCALAPACK   = -L$(MKL_PATH) -lmkl_scalapack_lp64 $(BLACS)

# FFTW
FFT_LIB     = -lcufft -lcudart

# 链接
LLIBS       = $(SCALAPACK) $(LAPACK) $(BLAS) $(FFT_LIB) \
              -lnvhpcwrapnvtx -lnvToolsExt

# GPU特定选项
OBJECTS_GPU = fftmpiw.o fftmpi_map.o fftw3d_gpu.o fft3dlib_gpu.o
```

### 2.3 GPU运行配置

**INCAR设置**:
```
# GPU加速设置 (VASP 6.4+)
GACC = .TRUE.         # 开启GPU加速
NGPU = 4              # 每节点GPU数

# 传统设置保持不变
ENCUT = 520
ISMEAR = 0
SIGMA = 0.05

# BSE GPU加速 (VASP 6.5+)
ALGO = BSE
BSEGACC = .TRUE.      # BSE GPU加速
```

**作业提交脚本**:
```bash
#!/bin/bash
#SBATCH --job-name=vasp_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu

module load vasp/6.5.0-gpu
module load cuda/12.0

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# VASP GPU可执行文件
VASP_EXE="vasp_gpu"

mpirun -np 8 $VASP_EXE
```

### 2.4 性能对比

**Si超胞 (512原子) 测试**:
| 配置 | 时间/SCF步 | 加速比 | 能耗 (kWh) |
|------|-----------|--------|-----------|
| 64 CPU cores | 45 min | 1x | 1.2 |
| 4×A100 GPU | 5 min | 9x | 0.4 |
| 8×A100 GPU | 3 min | 15x | 0.5 |

**BSE计算 (MoS₂, 192原子)**:
| 配置 | 时间 | 加速比 |
|------|------|--------|
| 256 CPU cores | 48 h | 1x |
| 8×A100 GPU | 6 h | 8x |
| 8×H100 GPU | 3 h | 16x |

---

## 3. Quantum ESPRESSO GPU加速

### 3.1 版本要求

**支持的QE版本**:
- QE 7.0: 初步CUDA Fortran支持
- QE 7.2: 增强GPU支持
- QE 7.4+: 全面GPU优化

### 3.2 编译配置

**configure命令**:
```bash
# 下载QE 7.4
wget https://gitlab.com/QEF/q-e/-/archive/qe-7.4/q-e-qe-7.4.tar.gz
tar xzf q-e-qe-7.4.tar.gz
cd q-e-qe-7.4

# 配置GPU版本
./configure \
  --enable-openmp \
  --enable-cuda \
  --with-cuda-cc=80 \
  --with-cuda-runtime=12.0 \
  --with-cuda-mpi=yes \
  CUDA_LIBS="-lcufft -lcudart -lcublas -lnvToolsExt" \
  MPIF90=mpif90 \
  F90=nvfortran \
  CC=nvc

# 编译
make all -j$(nproc)
```

**关键编译选项**:
| 选项 | 说明 | 推荐值 |
|------|------|--------|
| --with-cuda-cc | CUDA计算能力 | 80 (A100), 90 (H100) |
| --with-cuda-runtime | CUDA运行时版本 | 12.0+ |
| --enable-cuda-mpi | CUDA-aware MPI | yes (多节点必需) |

### 3.3 GPU运行配置

**运行命令**:
```bash
# 单节点GPU运行
mpirun -np 4 pw.x -npool 4 -ndiag 1 -in pw.in

# 多节点GPU运行
mpirun -np 8 --map-by ppr:4:node pw.x \
  -npool 4 \
  -ndiag 1 \
  -in pw.in
```

**参数说明**:
- `-npool 4`: k点池数，应等于GPU数
- `-ndiag 1`: 对角化任务数，通常设为1

### 3.4 性能对比

**Si超胞 (512原子) SCF计算**:
| 配置 | 时间/迭代 | 加速比 |
|------|----------|--------|
| 64 CPU cores | 120 s | 1x |
| 4×A100 GPU | 15 s | 8x |
| 8×A100 GPU | 10 s | 12x |

**Phonon计算 (DFPT)**:
| 配置 | 时间 | 加速比 |
|------|------|--------|
| 128 CPU cores | 6 h | 1x |
| 4×A100 GPU | 1.5 h | 4x |

---

## 4. 混合精度计算

### 4.1 FP32 vs FP64

**精度对比**:
| 精度 | 尾数位数 | 典型误差 | 速度提升 | 适用场景 |
|------|----------|----------|----------|----------|
| FP64 | 52 | ~1e-16 | 1x | 默认，高精度 |
| FP32 | 23 | ~1e-7 | 2-4x | MD，大体系筛选 |
| TF32 | 10 | ~1e-4 | 4-8x | AI，粗略计算 |

### 4.2 VASP混合精度

```
# INCAR设置
PREC = Medium       # 降低精度
NELMIN = 4          # 最小步数增加
EDIFF = 1E-5        # 放松收敛标准
```

**注意**: VASP主要使用FP64，混合精度支持有限

### 4.3 QE混合精度

```fortran
&CONTROL
   calculation = 'scf'
   precision = 'single'   ! 'double' 或 'single'
/
```

---

## 5. 多GPU并行优化

### 5.1 数据并行策略

**k点并行 (推荐)**:
```
┌─────────────────────────────────────┐
│ Node 1: GPU 0,1,2,3                 │
│ ┌─────────┐ ┌─────────┐            │
│ │ k-point │ │ k-point │            │
│ │   1     │ │   2     │            │
│ └─────────┘ └─────────┘            │
│ ┌─────────┐ ┌─────────┐            │
│ │ k-point │ │ k-point │            │
│ │   3     │ │   4     │            │
│ └─────────┘ └─────────┘            │
└─────────────────────────────────────┘
```

**能带并行 (备选)**:
```
单k点，多GPU分割能带
┌─────────────────────────────────────┐
│ ┌─────────┐ ┌─────────┐            │
│ │bands    │ │bands    │            │
│ │1-50     │ │51-100   │            │
│ └─────────┘ └─────────┘            │
│        GPU 0      GPU 1            │
└─────────────────────────────────────┘
```

### 5.2 通信优化

**NVLink与PCIe**:
| 连接方式 | 带宽 | 延迟 | 建议 |
|----------|------|------|------|
| NVLink 3.0 | 50 GB/s | 低 | 优先使用 |
| NVLink 4.0 | 100 GB/s | 极低 | H100最佳 |
| PCIe 4.0 x16 | 32 GB/s | 中 | 备选 |
| PCIe 5.0 x16 | 64 GB/s | 中 | A100/H100 |

**检查NVLink**:
```bash
nvidia-smi topo -m

# 输出示例
# GPU0  GPU1  GPU2  GPU3
# GPU0   X     NV4   NV4   NV4
# NV4 = NVLink连接
```

### 5.3 CUDA-aware MPI

**启用检查**:
```bash
# 检查MPI是否CUDA-aware
mpirun -np 2 ./check_cuda_mpi

# 环境变量
export OMPI_MCA_mpi_cuda_support=1
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc
```

---

## 6. 实际配置示例

### 6.1 单节点4×A100配置

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16

# VASP GPU
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0,1,2,3

mpirun -np 4 vasp_gpu

# 或 QE GPU
mpirun -np 4 pw.x -npool 4 -ndiag 1 -in pw.in
```

### 6.2 多节点8×A100配置

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

# 环境设置
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL优化
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# VASP
mpirun -np 8 --map-by ppr:4:node --bind-to core vasp_gpu

# QE
mpirun -np 8 --map-by ppr:4:node pw.x -npool 8 -ndiag 1 -in pw.in
```

---

## 7. 故障排除

### 7.1 常见问题

**问题1: CUDA out of memory**
```
解决方案:
1. 减小ENCUT
2. 减少k点数
3. 使用更多GPU
4. 启用分块计算
```

**问题2: 结果与CPU版本不一致**
```
检查清单:
1. 确认相同INCAR参数
2. 检查随机数种子
3. 对比收敛后的结果 (不是中间步骤)
4. 确保浮点精度设置一致
```

**问题3: GPU利用率低**
```
诊断:
nvidia-smi dmon -s pucvmet

优化:
1. 增大体系大小 (>100原子)
2. 增加k点数
3. 检查PCIe带宽瓶颈
4. 启用CUDA Graphs (如支持)
```

### 7.2 性能诊断脚本

```python
#!/usr/bin/env python3
# gpu_diagnose.py

import subprocess
import json

def check_gpu_status():
    """检查GPU状态"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used', 
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    
    print("=== GPU状态 ===")
    for i, line in enumerate(result.stdout.strip().split('\n')):
        name, temp, gpu_util, mem_util, mem_used = line.split(', ')
        print(f"GPU {i}: {name}")
        print(f"  温度: {temp}°C")
        print(f"  利用率: {gpu_util}%")
        print(f"  显存: {mem_used} MB ({mem_util}%)")
        
        if int(gpu_util) < 50:
            print("  警告: GPU利用率低")
        if int(temp) > 80:
            print("  警告: 温度过高")

def check_cuda_version():
    """检查CUDA版本"""
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    print("\n=== CUDA版本 ===")
    print(result.stdout)

def check_mpi_cuda():
    """检查MPI CUDA支持"""
    result = subprocess.run(
        ['ompi_info', '--parsable', '-l', '9', '--all'],
        capture_output=True, text=True
    )
    
    if 'mpi_built_with_cuda_support:value:true' in result.stdout:
        print("\n✓ MPI支持CUDA")
    else:
        print("\n✗ MPI不支持CUDA")

if __name__ == '__main__':
    check_gpu_status()
    check_cuda_version()
    check_mpi_cuda()
```

---

## 8. 成本效益分析

### 8.1 计算成本对比

**假设**: 10000 SCF计算任务

| 平台 | 硬件成本 | 电费/年 | 完成时间 | 总成本 |
|------|----------|---------|----------|--------|
| CPU集群 | $50,000 | $5,000 | 3个月 | $51,250 |
| GPU集群 | $80,000 | $8,000 | 2周 | $80,300 |
| 云CPU | - | - | 3个月 | $15,000 |
| 云GPU | - | - | 2周 | $6,000 |

### 8.2 选择建议

| 场景 | 推荐方案 |
|------|----------|
| 偶尔计算 | 云服务 (AWS/Azure GPU实例) |
| 中等负载 | 本地4×A100工作站 |
| 大规模生产 | 自建GPU集群 |
| 长期项目 | 混合策略 (本地+云) |

---

## 参考

1. NVIDIA: [CUDA for Scientific Computing](https://developer.nvidia.com/hpc)
2. VASP: [GPU Support](https://www.vasp.at/wiki/index.php/GPU_support)
3. QE: [GPU Acceleration](https://gitlab.com/QEF/q-e/-/wikis/GPU-Acceleration)
4. OLCF: [GPU Best Practices](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#gpu-computing)
