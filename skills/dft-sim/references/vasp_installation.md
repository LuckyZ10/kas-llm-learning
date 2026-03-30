# VASP 安装配置指南

## 系统要求

- Linux/Unix系统 (推荐CentOS 7+/Ubuntu 20.04+)
- Fortran/C/C++编译器 (Intel oneAPI 或 GNU)
- MPI库 (Intel MPI 或 OpenMPI)
- 数学库 (Intel MKL 或 OpenBLAS)
- FFT库 (FFTW 或 MKL FFT)

## 获取VASP

### 1. 申请许可证

访问 https://www.vasp.at/ 申请学术或商业许可证。

### 2. 下载源码

获得许可后，从VASP Portal下载：
- `vasp.6.5.0.tgz` (最新稳定版)
- `vasp.6.4.3.tgz` (上一稳定版)

## 安装步骤

### 方法一：使用Intel oneAPI (推荐)

```bash
# 1. 安装Intel oneAPI Base Toolkit和HPC Toolkit
# 下载地址: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

# 2. 设置环境变量
source /opt/intel/oneapi/setvars.sh

# 3. 解压VASP源码
tar -xzf vasp.6.5.0.tgz
cd vasp.6.5.0

# 4. 复制并修改makefile.include
cp arch/makefile.include.linux_intel makefile.include

# 5. 编辑makefile.include (关键配置)
# 以下是一个优化后的配置示例：
```

**makefile.include 示例 (Intel oneAPI)**:

```makefile
# Precompiler options
CPP_OPTIONS= -DHOST=\"LinuxIntel\" \
             -DMPI -DMPI_BLOCK=8000 -Duse_collective \
             -DscaLAPACK \
             -DCACHE_SIZE=4000 \
             -Davoidalloc \
             -Dvasp6 \
             -Duse_bse_te \
             -Dtbdyn \
             -Dfock_dblbuf \
             -D_OPENMP

CPP        = fpp -f_com=no -free -w0  $*$(FUFFIX) $*$(SUFFIX) $(CPP_OPTIONS)

FC         = mpiifx -qopenmp
FCL        = mpiifx -qmkl=parallel -qopenmp

FREE       = -free -names lowercase

FFLAGS     = -assume byterecl -w -xHOST
OFLAG      = -O2
OFLAG_IN   = $(OFLAG)
DEBUG      = -O0

MKL_PATH   = $(MKLROOT)/lib/intel64
BLAS       =
LAPACK     =
BLACS      =
SCALAPACK  =

OBJECTS    = fftmpiw.o fftmpi_map.o fftw3d.o fft3dlib.o

INCS       =-I$(MKLROOT)/include/fftw

LLIBS      = $(SCALAPACK) $(LAPACK) $(BLAS)

OBJECTS_O1 += fftw3d.o fftmpi.o fftmpiw.o
OBJECTS_O2 += fft3dlib.o

# For VASP.6.5.0+ with MLFF support
OBJECTS    += vaspml.o
LLIBS      += -L$(VASPML_PATH) -lvaspml

# HDF5 support (optional)
# CPP_OPTIONS += -DVASP_HDF5
# LLIBS       += -L$(HDF5_ROOT)/lib -lhdf5_fortran
# INCS        += -I$(HDF5_ROOT)/include
```

```bash
# 6. 编译
make std    # 标准版本
make gam    # Gamma点版本
make ncl    # 非共线版本

# 7. 安装
mkdir -p $HOME/bin/vasp
cp bin/vasp_std bin/vasp_gam bin/vasp_ncl $HOME/bin/vasp/

# 8. 添加到PATH
echo 'export PATH=$HOME/bin/vasp:$PATH' >> ~/.bashrc
```

### 方法二：使用GNU编译器

```bash
# 1. 安装依赖
# CentOS/RHEL
sudo yum install gcc gcc-gfortran gcc-c++ openmpi openmpi-devel fftw fftw-devel

# Ubuntu
sudo apt-get install gcc gfortran g++ libopenmpi-dev libfftw3-dev

# 2. 设置环境
module load openmpi

# 3. 复制GNU配置文件
cp arch/makefile.include.linux_gnu makefile.include

# 4. 编辑makefile.include，指定库路径
# 主要修改SCALAPACK、FFTW等库的路径

# 5. 编译
make std
```

### 方法三：GPU版本编译 (NVIDIA)

```bash
# 需要NVIDIA HPC SDK
module load nvhpc

cp arch/makefile.include.linux_nvidia makefile.include

# 编辑makefile.include启用OpenACC
# 添加: -acc -ta=tesla:cc80  (针对A100)

make std
```

## 验证安装

```bash
# 测试基本运行
mpirun -np 4 vasp_std

# 检查版本
vasp_std --version
```

## 常见问题

### 1. 编译错误 "undefined reference to"

**原因**: 库路径不正确
**解决**: 检查makefile.include中的LLIBS和INCS路径

### 2. "MKL not found"

**解决**: 
```bash
source /opt/intel/oneapi/setvars.sh
# 或在makefile.include中硬编码MKLROOT路径
```

### 3. 并行运行报错

**检查**: 
```bash
which mpirun
mpirun --version
# 确保使用与编译时一致的MPI
```

## 性能优化建议

1. **使用Intel编译器**: 通常比GNU快10-20%
2. **启用OpenMP**: 混合MPI+OpenMP可提高扩展性
3. **优化FFT**: 使用FFTW_MEASURE计划
4. **内存分配**: 大体系考虑使用`-Davoidalloc`

## VASP 6.5 新特性配置

### 启用机器学习力场 (MLFF)

```bash
# 编译时需要包含VASPml库
# 在makefile.include中:
OBJECTS += vaspml.o
LLIBS   += -L/path/to/vaspml -lvaspml
```

### 启用HDF5输出

```makefile
CPP_OPTIONS += -DVASP_HDF5
LLIBS       += -L$(HDF5_ROOT)/lib -lhdf5_fortran -lhdf5
INCS        += -I$(HDF5_ROOT)/include
```

### Python插件支持

```bash
# 需要Python 3.8+和pybind11
pip install pybind11
# 编译时自动检测并启用
```

## 参考

- VASP安装手册: https://www.vasp.at/wiki/index.php/Installing_VASP
- Intel oneAPI: https://www.intel.com/oneapi
