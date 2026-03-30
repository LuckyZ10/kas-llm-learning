# Quantum ESPRESSO 安装配置指南

## 系统要求

- Linux/Unix/macOS系统
- Fortran 2008兼容编译器 (gfortran/ifx/ifort)
- C/C++编译器
- MPI库 (OpenMPI/Intel MPI/MPICH)
- 数学库 (BLAS/LAPACK/ScaLAPACK)
- FFT库 (FFTW或MKL)
- CMake 3.14+ 或 GNU Make

## 下载Quantum ESPRESSO

```bash
# 方法1: 从GitLab下载最新稳定版 (推荐)
wget https://gitlab.com/QEF/q-e/-/archive/qe-7.4/q-e-qe-7.4.tar.gz
tar -xzf q-e-qe-7.4.tar.gz
cd q-e-qe-7.4

# 方法2: 从官网下载
wget https://www.quantum-espresso.org/rdm-download/488/qe-7.4-ReleasePack.tar.gz

# 方法3: 使用git克隆开发版
git clone https://gitlab.com/QEF/q-e.git
cd q-e
```

## 安装方法

### 方法一：使用CMake (推荐，7.0+)

CMake提供更灵活的构建选项，特别是GPU支持。

#### 基础安装

```bash
cd q-e-qe-7.4
mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=/path/to/install \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DCMAKE_C_COMPILER=mpicc \
  -DQE_ENABLE_MPI=ON \
  -DQE_ENABLE_OPENMP=ON

make -j$(nproc)
make install
```

#### 使用Intel oneAPI优化

```bash
# 加载Intel环境
source /opt/intel/oneapi/setvars.sh

mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=/opt/qe/7.4-intel \
  -DCMAKE_Fortran_COMPILER=mpiifx \
  -DCMAKE_C_COMPILER=mpiicx \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DMKL_ROOT=$MKLROOT \
  -DQE_ENABLE_MPI=ON \
  -DQE_ENABLE_OPENMP=ON \
  -DQE_ENABLE_SCALAPACK=ON

make -j$(nproc)
make install
```

#### GPU加速版本 (CUDA)

```bash
# 需要NVIDIA HPC SDK
module load nvhpc

mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=/opt/qe/7.4-gpu \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DQE_ENABLE_CUDA=ON \
  -DQE_ENABLE_OPENACC=ON \
  -DNVFORTRAN_CUDA_CC=80 \
  -DNVFORTRAN_CUDA_VERSION=12.6 \
  -DQE_ENABLE_MPI_GPU_AWARE=ON \
  -DQE_ENABLE_OPENMP=ON

make -j$(nproc)
make install
```

**关键CUDA选项说明**:
- `NVFORTRAN_CUDA_CC`: GPU计算能力 (70=V100, 80=A100, 90=H100)
- `QE_ENABLE_MPI_GPU_AWARE`: 启用CUDA-aware MPI (需要NVLink或InfiniBand)

### 方法二：使用传统configure/make

```bash
cd q-e-qe-7.4

# 基础配置
./configure --prefix=/opt/qe/7.4

# 启用OpenMP
./configure --enable-openmp --prefix=/opt/qe/7.4

# 指定编译器
./configure MPIF90=mpif90 F90=gfortran CC=gcc --prefix=/opt/qe/7.4

# 编译
make all -j$(nproc)
make install
```

#### configure常用选项

| 选项 | 说明 |
|------|------|
| `--enable-parallel` | 启用MPI并行 (默认) |
| `--enable-openmp` | 启用OpenMP支持 |
| `--enable-static` | 静态链接 |
| `--with-scalapack` | 使用ScaLAPACK |
| `--with-elpa` | 使用ELPA对角化库 |
| `--with-hdf5` | 启用HDF5支持 |
| `--with-libxc` | 使用libxc泛函库 |
| `--with-cuda` | 启用CUDA (GPU) |

### 方法三：使用Spack包管理器

```bash
# 安装Spack (如果尚未安装)
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh

# 安装QE
spack install quantum-espresso@7.4

# 加载QE
spack load quantum-espresso@7.4
```

## 配置环境变量

```bash
# 添加到 ~/.bashrc 或 ~/.bash_profile

# QE安装路径
export QE_HOME=/opt/qe/7.4
export PATH=$QE_HOME/bin:$PATH

# 伪势库路径
export ESPRESSO_PSEUDO=$QE_HOME/pseudo

# OpenMP线程数
export OMP_NUM_THREADS=1

# MPI环境 (如需要)
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

## 下载伪势库

```bash
# 方法1: 自动下载 (运行测试时)
make test  # 会自动下载部分伪势

# 方法2: 手动下载标准赝势库
mkdir -p $ESPRESSO_PSEUDO
cd $ESPRESSO_PSEUDO

# SSSP (Standard Solid State Pseudopotentials)
wget https://www.materialscloud.org/sssp/1.3.0/pbe/SSSP_1.3.0_PBE_efficiency.tar.gz
tar -xzf SSSP_1.3.0_PBE_efficiency.tar.gz

# GBRV赝势
wget http://www.physics.rutgers.edu/gbrv/all_pbe_UPFv1.5.tar.gz
tar -xzf all_pbe_UPFv1.5.tar.gz

# Dojo赝势
wget http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard.tar.gz
tar -xzf nc-sr-04_pbe_standard.tar.gz
```

## 验证安装

```bash
# 测试串行版本
pw.x -in test.in

# 测试并行版本
mpirun -np 4 pw.x -in test.in

# 运行测试套件
cd build
make test

# 或运行特定测试
ctest -R pw_scf -V
```

## 常见问题解决

### 1. "configure: error: cannot find Fortran compiler"

```bash
# 确保编译器在PATH中
which gfortran
which mpif90

# 如使用模块系统
module load gcc openmpi
```

### 2. "undefined reference to 'mpi_init_'"

**原因**: MPI库链接问题
**解决**:
```bash
# 检查MPI Fortran包装器
mpif90 --showme

# 重新配置，显式指定
./configure MPIF90=/usr/lib/openmpi/bin/mpif90
```

### 3. 编译时内存不足

```bash
# 减少并行编译任务数
make -j2  # 而非 -j$(nproc)

# 或分段编译
make pw  # 先编译PWscf
make ph  # 再编译PHonon
```

### 4. GPU版本运行失败

```bash
# 检查CUDA环境
cuda-smi
nvcc --version

# 确保CUDA计算能力匹配
# 编辑cmake配置，设置正确的NVFORTRAN_CUDA_CC
```

## 性能优化

### 1. 混合MPI+OpenMP

```bash
# 设置环境变量
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# 运行
mpirun -np 8 --bind-to socket pw.x -in input.in
```

### 2. k点并行优化

```bash
# 在输入文件中设置
&SYSTEM
  kpar = 4  ! k点组数，应≤MPI进程数
/
```

### 3. 使用ELPA加速对角化

```bash
cmake .. -DQE_ENABLE_ELPA=ON \
         -DELPA_ROOT=/path/to/elpa
```

## QE 7.4 新特性

- **GPU全面支持**: CUDA + OpenACC优化
- **双化学势声子**: 用于热电材料计算
- **改进的对称性检测**: 自动识别更多空间群
- **性能提升**: 相比7.3版本，CPU版本提升10-20%

## 参考资源

- QE用户指南: https://www.quantum-espresso.org/Doc/user_guide/
- QE论坛: https://www.quantum-espresso.org/forum/
- GitLab仓库: https://gitlab.com/QEF/q-e
