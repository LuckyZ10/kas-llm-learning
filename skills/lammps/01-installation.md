# 01. LAMMPS 安装配置

## 目录
- [快速安装](#快速安装)
- [源码编译详解](#源码编译详解)
- [并行配置](#并行配置)
- [GPU加速](#gpu加速)
- [常用用户包](#常用用户包)
- [验证测试](#验证测试)

---

## 快速安装

### Ubuntu/Debian 系统包
```bash
# 基础安装
sudo apt-get update
sudo apt-get install lammps

# 完整安装（含额外包）
sudo apt-get install lammps-stable
```

### Conda 安装（推荐用户环境）
```bash
# 创建环境
conda create -n lammps_env python=3.11
conda activate lammps_env

# 安装LAMMPS
conda install -c conda-forge lammps

# 安装额外包
conda install -c conda-forge lammps-dp  # DeepMD支持
```

### 预编译二进制
```bash
# 下载稳定版本
wget https://download.lammps.org/tars/lammps-stable.tar.gz
tar -xzf lammps-stable.tar.gz
cd lammps-*/src

# 预编译可执行文件（部分包）
make yes-standard
make mpi  # 或 make serial
```

---

## 源码编译详解

### 1. 依赖安装

```bash
# 基础编译工具
sudo apt-get install build-essential cmake git

# MPI库（并行）
sudo apt-get install openmpi-bin libopenmpi-dev
# 或 MPICH
sudo apt-get install mpich libmpich-dev

# FFTW3（长程力计算）
sudo apt-get install libfftw3-dev

# HDF5（NetCDF支持）
sudo apt-get install libhdf5-dev

# Python绑定（可选）
sudo apt-get install python3-dev python3-pip
pip install numpy

# JPEG/PNG支持（dump图像）
sudo apt-get install libjpeg-dev libpng-dev

# CUDA（GPU支持）
# 需从NVIDIA官网安装对应版本的CUDA Toolkit
```

### 2. 传统Makefile编译

```bash
cd lammps/src

# 启用标准包
make yes-standard

# 启用特定包
make yes-molecule      # 分子力场
make yes-kspace        # 长程库仑
make yes-manybody      # 多体势
make yes-rigid         # 刚体约束
make yes-reaxff        # ReaxFF反应力场
make yes-meam          # MEAM势
make yes-mlip          # MLIP机器学习势
make yes-pace          # ACE势
make yes-deepmd        # DeepMD-kit

# 编译并行版本
make mpi -j$(nproc)

# 编译串行版本
make serial -j$(nproc)

# 输出可执行文件: lmp_mpi 或 lmp_serial
```

### 3. CMake编译（推荐）

```bash
mkdir -p lammps/build
cd lammps/build

# 基础配置
cmake ../cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_MPI=yes \
    -DBUILD_OMP=yes \
    -DPKG_MOLECULE=yes \
    -DPKG_KSPACE=yes \
    -DPKG_MANYBODY=yes \
    -DPKG_RIGID=yes

# 高级配置（含GPU和机器学习）
cmake ../cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_MPI=yes \
    -DBUILD_OMP=yes \
    -DPKG_MOLECULE=yes \
    -DPKG_KSPACE=yes \
    -DPKG_MANYBODY=yes \
    -DPKG_REAXFF=yes \
    -DPKG_ML-PACE=yes \
    -DPKG_ML-POD=yes \
    -DPKG_KOKKOS=yes \
    -DKokkos_ENABLE_CUDA=yes \
    -DKokkos_ARCH_AMPERE80=yes \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# 编译
make -j$(nproc)

# 安装
sudo make install
```

### 4. Python接口安装

```bash
cd lammps/python
pip install .

# 验证
python -c "from lammps import lammps; lmp = lammps()"
```

---

## 并行配置

### MPI并行

```bash
# 编译MPI版本
cd lammps/src
make yes-standard
make mpi -j$(nproc)

# 运行MPI并行
mpirun -np 8 ./lmp_mpi -in input.lammps

# 多节点运行
mpirun -np 64 --hostfile hosts.txt ./lmp_mpi -in input.lammps

# 绑定核心（性能优化）
mpirun -np 8 --bind-to core ./lmp_mpi -in input.lammps
```

### OpenMP混合并行

```bash
# 编译OpenMP支持
cmake ../cmake -DBUILD_OMP=yes

# 设置线程数
export OMP_NUM_THREADS=4

# 混合MPI+OpenMP运行
mpirun -np 4 ./lmp -sf omp -in input.lammps
```

### MPI配置建议

| 系统规模 | MPI进程数 | OpenMP线程 | 总核心数 |
|---------|----------|-----------|---------|
| <10K原子 | 1-4 | 1 | 1-4 |
| 10K-100K | 4-16 | 1-2 | 4-32 |
| 100K-1M | 16-64 | 2-4 | 32-256 |
| >1M原子 | 64-1024 | 4-8 | 256-8K |

---

## GPU加速

### Kokkos GPU加速

```bash
# CMake配置GPU
cmake ../cmake \
    -DPKG_KOKKOS=yes \
    -DKokkos_ENABLE_CUDA=yes \
    -DKokkos_ARCH_AMPERE80=yes \
    -DKokkos_ENABLE_OPENMP=yes

# 运行时GPU加速
lmp -k on g 1 -sf kk -in input.lammps    # 1 GPU
lmp -k on g 4 -sf kk -in input.lammps    # 4 GPU

# 混合MPI+GPU
mpirun -np 4 lmp -k on g 1 -sf kk -in input.lammps  # 4 MPI × 1 GPU
```

### 传统GPU包（已弃用，建议使用Kokkos）

```bash
# 启用GPU包
make yes-gpu

# 编译
make mpi -j$(nproc)

# 运行
lmp -sf gpu -pk gpu 1 -in input.lammps
```

### GPU性能优化技巧

```ini
# input.lammps - GPU优化设置
# 1. 使用neigh_modify减少邻居列表重建
neigh_modify every 1 delay 0 check yes page 100000 one 10000

# 2. 使用newton off提高GPU效率
newton off

# 3. 选择合适的pair_style
pair_style lj/cut/gpu 10.0          # GPU版本
pair_style lj/cut/kk 10.0           # Kokkos版本
```

---

## 常用用户包

### 标准包启用指南

```bash
cd lammps/src

# 分子模拟
make yes-molecule      # 分子力场(topology, bonds, angles...)
make yes-amber         # AMBER力场支持
make yes-charmm        # CHARMM力场支持
make yes-class2        # COMPASS力场

# 机器学习势
make yes-ml-pace       # ACE势
make yes-ml-pod        # POD势
make yes-ml-rann       # RANN势
make yes-deepmd        # DeepMD-kit接口

# 反应力场
make yes-reaxff        # ReaxFF
make yes-reaction      # 反应化学

# 高级采样
make yes-colvars       #  collective variables
make yes-plumed        # PLUMED接口
make yes-hdf5          # HDF5输出

# 其他重要包
make yes-voronoi       # Voronoi分析
make yes-atomify       # 可视化工具
make yes-latte         # LATTE紧束缚
make yes-mdi           # MDI接口
```

### 常用包说明表

| 包名 | 功能 | 典型应用 |
|-----|------|---------|
| MOLECULE | 分子力场 | 有机分子、生物分子 |
| KSPACE | 长程库仑 | 离子体系、晶体 |
| MANYBODY | 多体势 | 金属、半导体 |
| RIGID | 刚体约束 | 水模型、刚体分子 |
| REAXFF | 反应力场 | 化学反应模拟 |
| COLVARS | 集体变量 | 自由能计算 |
| PLUMED | 高级采样 | Metadynamics |
| DEEPMD | 深度势 | 机器学习MD |
| MEAM | 修正EAM | 合金体系 |
| OPENMP | OpenMP并行 | 多核加速 |

---

## 验证测试

### 基本功能测试

```bash
# 测试串行版本
cd lammps/examples/melt
../../src/lmp_serial -in in.melt

# 测试并行版本
mpirun -np 4 ../../src/lmp_mpi -in in.melt

# 测试GPU版本
../../src/lmp -k on g 1 -sf kk -in in.melt
```

### 常见安装问题

```bash
# 问题1: mpi.h not found
sudo apt-get install libopenmpi-dev

# 问题2: fftw3.h not found
sudo apt-get install libfftw3-dev

# 问题3: undefined reference to `omp_get_thread_num'
export LDFLAGS="-fopenmp"

# 问题4: CUDA架构不匹配
# 修改cmake中的 Kokkos_ARCH_AMPERE80 为对应GPU架构
# Pascal: SM60, SM61
# Volta: SM70
# Turing: SM75
# Ampere: SM80, SM86
# Hopper: SM90

# 问题5: Python绑定失败
export PYTHONPATH=/path/to/lammps/python:$PYTHONPATH
```

### 性能基准测试

```bash
# 下载Lennard-Jones melt测试
cd lammps/bench

# 运行基准
mpirun -np 8 ../src/lmp_mpi -in in.lj

# 查看输出: 原子数/秒 或 纳秒/天
```

---

## 环境变量配置

```bash
# .bashrc 或 .zshrc 添加
export PATH=/path/to/lammps/src:$PATH
export LAMMPS_POTENTIALS=/path/to/lammps/potentials
export PYTHONPATH=/path/to/lammps/python:$PYTHONPATH

# OpenMP线程设置
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# MPI设置
export OMPI_MCA_btl_vader_single_copy_mechanism=none  # OpenMPI
```

---

## 参考命令速查

```bash
# 查看可用包
make package-status

# 清理编译
make clean-all
make clean-machine

# 查看帮助
./lmp -h

# 查看可用accelerator styles
./lmp -h | grep -i "accelerator"
```
