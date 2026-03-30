# 08. 性能优化与并行计算

> LAMMPS高性能计算、GPU加速和扩展性优化指南

---

## 目录
- [并行架构概述](#并行架构概述)
- [MPI并行优化](#mpi并行优化)
- [OpenMP混合并行](#openmp混合并行)
- [GPU加速](#gpu加速)
- [Kokkos编程模型](#kokkos编程模型)
- [负载均衡](#负载均衡)
- [IO优化](#io优化)
- [基准测试与调优](#基准测试与调优)

---

## 并行架构概述

### 1. LAMMPS并行层次

```
LAMMPS并行层次:
├── MPI (分布式内存)
│   ├── 空间分解 (Domain Decomposition)
│   └── 复制分解 (Replication)
├── OpenMP (共享内存)
│   └── 循环并行
├── GPU (加速卡)
│   ├── Kokkos CUDA
│   └── OpenCL/ROC
└── 混合模式
    └── MPI + OpenMP + GPU
```

### 2. 并行选择指南

| 体系规模 | 推荐并行模式 | 进程/线程配置 |
|---------|-------------|--------------|
| < 10K原子 | 串行/OpenMP | 1 MPI × 4-8线程 |
| 10K-100K | MPI | 4-16 MPI |
| 100K-1M | MPI | 16-64 MPI |
| 1M-10M | MPI(+OpenMP) | 64-256 MPI |
| > 10M | 混合MPI+OpenMP | 256+ MPI × 4线程 |
| GPU可用 | Kokkos | 1-8 GPU |

---

## MPI并行优化

### 1. 空间分解策略

```lammps
# 默认空间分解
mpirun -np 64 lmp -in input.lammps

# 指定分解方式
mpirun -np 64 lmp -in input.lammps -partition 4x4x4

# 自动负载均衡
fix 1 all balance 1000 shift xyz 10 1.1
```

### 2. 邻居列表优化

```lammps
# 优化邻居列表构建
neighbor 2.0 bin          # 皮肤距离
neigh_modify every 1 delay 0 check yes

# 大体系优化
neigh_modify every 1 delay 0 check yes page 100000 one 10000

# 减少邻居列表重建频率 (性能vs精度权衡)
neigh_modify every 10 delay 0 check yes
```

### 3. 通信优化

```lammps
# 半精度通信 (双精度计算，单精度通信)
atom_style atomic
newton on               # 开启通信优化

# 减少通信
comm_style tiled        # 瓦片通信 (非均匀体系)
comm_modify mode single cutoff 12.0  # 单一通信截断
```

### 4. MPI运行优化

```bash
# OpenMPI绑定
mpirun -np 64 --bind-to core --map-by core lmp -in input.lammps

# 多节点
mpirun -np 256 --hostfile hosts.txt --bind-to socket lmp -in input.lammps

# 环境变量优化
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl_vader_flags=0

# Intel MPI
export I_MPI_PIN_DOMAIN=socket
export I_MPI_PIN_ORDER=compact
mpirun -np 64 -ppn 32 lmp -in input.lammps
```

---

## OpenMP混合并行

### 1. 编译配置

```bash
# CMake配置
cmake ../cmake \
    -DBUILD_OMP=yes \
    -DOpenMP_CXX_FLAGS="-fopenmp" \
    -DOpenMP_CXX_LIB_NAMES="omp"

# Makefile
make yes-openmp
make omp -j$(nproc)
```

### 2. 输入脚本设置

```lammps
# 启用OpenMP后缀
suffix omp

# 或在运行时使用
mpirun -np 4 lmp -in input.lammps -sf omp
```

### 3. 环境变量

```bash
# 设置线程数
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# 混合运行
mpirun -np 4 -x OMP_NUM_THREADS=4 lmp -in input.lammps -sf omp
```

### 4. 性能调优

```lammps
# 最佳实践
timestep 1.0

# 使用OpenMP优化的pair_style
pair_style lj/cut/omp 10.0
pair_style eam/alloy/omp

# kspace并行
kspace_style pppm/omp 1.0e-4
```

---

## GPU加速

### 1. Kokkos GPU配置

```bash
# CMake配置CUDA Kokkos
cmake ../cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_MPI=yes \
    -DPKG_KOKKOS=yes \
    -DKokkos_ENABLE_CUDA=yes \
    -DKokkos_ARCH_AMPERE80=yes \  # 根据GPU架构调整
    -DKokkos_ENABLE_OPENMP=yes \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# 支持的GPU架构
# Pascal:   Kokkos_ARCH_PASCAL60, Kokkos_ARCH_PASCAL61
# Volta:    Kokkos_ARCH_VOLTA70, Kokkos_ARCH_VOLTA72
# Turing:   Kokkos_ARCH_TURING75
# Ampere:   Kokkos_ARCH_AMPERE80, Kokkos_ARCH_AMPERE86
# Hopper:   Kokkos_ARCH_HOPPER90
```

### 2. GPU运行

```lammps
# 基本GPU运行
lmp -k on g 1 -sf kk -in input.lammps    # 1 GPU
lmp -k on g 4 -sf kk -in input.lammps    # 4 GPU

# 多MPI进程+GPU
mpirun -np 4 lmp -k on g 1 -sf kk -in input.lammps  # 4 MPI × 1 GPU
mpirun -np 8 lmp -k on g 2 -sf kk -in input.lammps  # 8 MPI × 2 GPU

# 显式设备选择
lmp -k on g 1 -sf kk -pk kokkos cuda/aware on -in input.lammps
```

### 3. GPU优化设置

```lammps
# GPU优化的输入脚本
units metal
atom_style atomic

# 使用Kokkos风格
pair_style lj/cut/kk 10.0    # Kokkos版本
pair_style eam/alloy/kk      # Kokkos版本

# 邻居列表优化 (GPU)
neigh_modify every 1 delay 0 check yes page 100000 one 10000

# 关闭newton (GPU性能更好)
newton off

# 使用GPU优化的kspace
kspace_style pppm/kk 1.0e-4

# 积分器
timestep 0.001

# 热浴
fix 1 all nvt/kk temp 300.0 300.0 0.1  # Kokkos版本

run 10000
```

### 4. GPU性能监控

```bash
# 监控GPU使用率
nvidia-smi -l 1

# 详细性能分析
nsys profile -o profile mpirun -np 1 lmp -k on g 1 -sf kk -in input.lammps

# Nsight Compute
ncu -o profile.ncu-rep lmp -k on g 1 -sf kk -in input.lammps
```

---

## Kokkos编程模型

### 1. Kokkos后端选择

```bash
# 运行时选择后端
export KOKKOS_DEVICE="Cuda"      # CUDA
export KOKKOS_DEVICE="OpenMP"    # OpenMP
export KOKKOS_DEVICE="HIP"       # AMD GPU
export KOKKOS_DEVICE="SYCL"      # Intel GPU
```

### 2. 高级Kokkos选项

```lammps
# 显存优化
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in input.lammps

# 多GPU节点
lmp -k on g 2 -sf kk -pk kokkos numa 2 -in input.lammps

# 异步执行
lmp -k on g 1 -sf kk -pk kokkos exchange comm host -in input.lammps
```

### 3. 设备内存管理

```lammps
# 优化数据移动
atom_style atomic/kk  # Kokkos原子风格

# 减少CPU-GPU数据传输
fix 1 all nvt/kk temp 300.0 300.0 0.1
```

---

## 负载均衡

### 1. 动态负载均衡

```lammps
# 每1000步重新平衡
fix 1 all balance 1000 shift xyz 10 1.1

# 基于粒子的负载均衡
fix 1 all balance 1000 shift xyz 10 1.1 weight time 1.0

# 使用rcb (递归坐标二分)
fix 1 all balance 1000 rcb

# 仅平衡特定维度
fix 1 all balance 1000 shift x 10 1.1
```

### 2. 静态分区

```lammps
# 启动时指定分区
# mpirun -np 8 lmp -in input.lammps -partition 2x2x2

# 输入脚本中指定
partition yes 1 2 2
```

### 3. 不均匀体系

```lammps
# 液滴在表面 (密度不均)
fix 1 all balance 1000 shift xyz 10 1.1 weight stored 1.0

# 多相体系
fix 1 all balance 1000 shift xyz 10 1.1 weight time 1.0 weight neigh 0.5
```

---

## IO优化

### 1. Dump优化

```lammps
# 减少IO频率
dump 1 all custom 10000 dump.lammpstrj id type x y z

# 压缩输出
dump 1 all custom 1000 dump.gz id type x y z

# 并行NetCDF
dump 1 all netcdf 1000 dump.nc id type x y z

# 仅输出特定原子
dump 1 surface custom 1000 surface.dump id type x y z
```

### 2. 二进制重启

```lammps
# 二进制格式 (更快)
write_restart binary.restart
read_restart binary.restart

# 并行写入 (多文件)
write_restart binary.%.*.restart
```

### 3. 减少日志输出

```lammps
# 减少热力学输出频率
thermo 10000

# 仅输出关键量
thermo_style custom step temp pe press

# 最小化输出
thermo_modify line one format float %g
```

---

## 基准测试与调优

### 1. 标准基准测试

```bash
# 下载官方基准
cd lammps/bench

# 运行基准
mpirun -np 64 lmp -in in.lj
mpirun -np 64 lmp -in in.eam
mpirun -np 64 lmp -in in.chute
mpirun -np 64 lmp -in in.rhodo

# 记录性能
tail -5 log.lammps
```

### 2. 扩展性测试

```bash
#!/bin/bash
# scaling_test.sh

input_file="benchmark.in"
max_cores=64

for n in 1 2 4 8 16 32 64; do
    echo "Running with $n cores..."
    mpirun -np $n lmp -in $input_file -log log.$n
    
    # 提取性能
    perf=$(grep "Loop time" log.$n | awk '{print $6}')
    echo "$n $perf" >> scaling.dat
done

# 绘图
python plot_scaling.py
```

```python
# plot_scaling.py
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('scaling.dat')
cores = data[:, 0]
time = data[:, 1]

# 计算加速比
speedup = time[0] / time
efficiency = speedup / cores

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(cores, speedup, 'bo-', label='Actual')
axes[0].plot(cores, cores, 'k--', label='Ideal')
axes[0].set_xlabel('Number of Cores')
axes[0].set_ylabel('Speedup')
axes[0].set_title('Strong Scaling')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(cores, efficiency * 100, 'ro-')
axes[1].axhline(y=100, color='k', linestyle='--')
axes[1].set_xlabel('Number of Cores')
axes[1].set_ylabel('Efficiency (%)')
axes[1].set_title('Parallel Efficiency')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('scaling.png', dpi=300)
```

### 3. 性能分析

```bash
# Intel VTune
vtune -collect hotspots -result-dir vtune_results mpirun -np 4 lmp -in input.lammps

# Intel Advisor (向量化分析)
advisor -collect survey -project-dir advisor_results mpirun -np 1 lmp -in input.lammps

# Perf (Linux)
perf record -g mpirun -np 1 lmp -in input.lammps
perf report

# HPCToolkit
hpcrun -t -e REALTIME mpirun -np 4 lmp -in input.lammps
hpcstruct lmp
hpcprof -S lmp.hpcstruct hpctoolkit-lmp-measurements
```

### 4. 调优检查清单

```
性能调优检查清单:

编译时:
□ 使用Release模式 (-O3)
□ 启用MPI支持
□ 启用OpenMP (如需要)
□ 启用Kokkos/GPU (如需要)
□ 优化FFT库 (FFTW3/MKL)

运行时:
□ 合适的MPI进程数
□ 启用OpenMP混合并行
□ 优化邻居列表参数
□ 使用负载均衡
□ 减少IO频率
□ 压缩输出

GPU专用:
□ 正确的GPU架构
□ 启用CUDA感知MPI
□ 优化newton设置
□ 检查显存使用
□ 异步数据传输
```

---

## 高级优化技巧

### 1. 多时间步长RESPA

```lammps
# 分层积分策略
run_style respa 4 2 2 2 inner 2 4.0 6.0 middle 4 8.0 10.0 outer 12.0

# 分解:
# Level 1: 键/角/近程 (inner) - 每2步
# Level 2: 中程 (middle) - 每4步
# Level 3: 长程kspace (outer) - 每8步
```

### 2. 表格式势函数

```lammps
# 使用表格加速
pair_style table linear 1000
pair_coeff 1 1 lj.table LJ

# 或使用pair_write生成
pair_write 1 1 1000 r 1.0 10.0 lj.txt LJ
```

### 3. 单精度模式

```lammps
# 混合精度 (计算双精度，通信单精度)
atom_style atomic
newton on

# 编译时启用混合精度
-DLAMMPS_SINGLE_PRECISION
```

---

## 性能参考数据

### 典型体系性能 (原子数/秒)

| 体系 | 1核 | 64核 | 1 GPU (A100) | 扩展性 |
|-----|-----|------|-------------|-------|
| LJ (1M原子) | 50K | 2M | 10M | 好 |
| EAM (1M原子) | 30K | 1.5M | 8M | 好 |
| ReaxFF (100K) | 2K | 100K | 500K | 中等 |
| ML势 (10K) | 100 | 5K | 50K | 有限 |

### 内存需求估算

```
内存/原子 ≈ 100-500 bytes (取决于模型)

示例:
- 100万原子: 100-500 MB
- 1000万原子: 1-5 GB
- 1亿原子: 10-50 GB
```

---

## 故障排除

### 1. 并行错误

```bash
# 死锁问题
export OMPI_MCA_mpi_yield_when_idle=1

# 内存不足
ulimit -s unlimited
export OMP_STACKSIZE=1G

# 通信错误
export OMPI_MCA_btl=^openib  # 禁用InfiniBand
```

### 2. GPU错误

```bash
# CUDA内存不足
# 减少每GPU的原子数或降低邻居列表page大小

# 驱动问题
nvidia-smi  # 检查GPU状态
nvcc --version  # 检查CUDA版本
```

### 3. 性能下降原因

```
常见性能问题:
1. 负载不均衡 → 使用fix balance
2. 邻居列表频繁重建 → 增大skin距离
3. IO瓶颈 → 减少dump频率，使用二进制
4. 通信开销 → 优化MPI拓扑
5. 内存带宽限制 → 启用矢量化
```

---

## 参考资源

- [LAMMPS Performance Guide](https://docs.lammps.org/Speed.html)
- [Kokkos Documentation](https://kokkos.github.io/kokkos-core-wiki/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [MPI Performance Tuning](https://www.open-mpi.org/faq/?category=perftools)
