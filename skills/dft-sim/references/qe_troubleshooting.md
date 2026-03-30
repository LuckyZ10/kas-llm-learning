# Quantum ESPRESSO常见错误排查指南

## 1. 编译错误

### 错误: "configure: error: cannot find Fortran compiler"
**解决**:
```bash
# 检查编译器
which gfortran
which mpif90

# 安装或加载
module load gcc openmpi

# 或指定路径
./configure MPIF90=/usr/bin/mpif90
```

### 错误: "libmkl_scalapack_lp64.so: cannot open"
**解决**:
```bash
# 加载MKL环境
source /opt/intel/oneapi/setvars.sh

# 或禁用ScaLAPACK
./configure --with-scalapack=no
```

### 错误: "wannier90 not found"
**解决**:
```bash
# 下载wannier90并放在archive目录
wget https://github.com/wannier-developers/wannier90/archive/v3.1.0.tar.gz
mv v3.1.0.tar.gz q-e-7.4/archive/wannier90-3.1.0.tar.gz
```

## 2. 运行时错误

### 错误: "Error in routine c_bands: electrons not converged"
**原因**: SCF不收敛
**解决**:
```fortran
&ELECTRONS
  conv_thr = 1.0D-7       ! 放宽收敛标准
  mixing_beta = 0.3       ! 降低混合参数
  electron_maxstep = 200  ! 增加最大步数
  mixing_mode = 'local-TF' ! 改变混合模式
/
```

### 错误: "charge density is not correct"
**原因**: 初始猜测问题
**解决**:
```fortran
&ELECTRONS
  startingpot = 'atomic'  ! 或 'file'
  startingwfc = 'atomic'  ! 或 'file'
/
```

### 错误: "symmetry operation is non-orthogonal"
**原因**: 结构对称性问题
**解决**:
```fortran
&SYSTEM
  nosym = .true.          ! 关闭对称性
  noinv = .true.          ! 关闭反演
/
```

### 错误: "atoms overlap!"
**原因**: 原子位置重叠
**解决**: 检查ATOMIC_POSITIONS，确保原子间距合理

## 3. 收敛问题

### SCF振荡
```fortran
&ELECTRONS
  ! 方法1: 调整展宽
  occupations = 'smearing'
  smearing = 'gaussian'   ! 或 'mp', 'mv'
  degauss = 0.01          ! 减小展宽
  
  ! 方法2: 调整混合
  mixing_beta = 0.3
  mixing_ndim = 12
  
  ! 方法3: 使用Pulay混合
  mixing_mode = 'plain'   ! 或 'TF', 'local-TF'
/
```

### 离子弛豫不收敛
```fortran
&CONTROL
  nstep = 200             ! 增加步数
/
&IONS
  ion_dynamics = 'bfgs'   ! 或 'damp'
  pot_extrapolation = 'second_order'
  wfc_extrapolation = 'second_order'
/
```

## 4. 并行错误

### 错误: "mpirun could not find executable"
**解决**:
```bash
# 检查PATH
which pw.x

# 或使用完整路径
mpirun -np 4 /opt/qe/7.4/bin/pw.x -in input.in
```

### 错误: "wrong number of MPI processes"
**原因**: k点并行设置问题
**解决**:
```fortran
&SYSTEM
  kpar = 2                ! 确保kpar ≤ MPI进程数
/
```

### 错误: "out of memory"
**解决**:
```bash
# 增加进程数 (减少每进程内存)
mpirun -np 16 pw.x -in input.in

# 或降低截断能
ecutwfc = 30              ! 临时降低
```

## 5. 声子计算错误

### 错误: "phq_readin: DFT-D3 not available"
**原因**: 声子计算不支持DFT-D3
**解决**:
```fortran
&INPUTPH
  ! 移除或注释掉DFT-D3相关设置
  ! dftd3 = .true.
/
```

### 错误: "phq_setup: lrpa not implemented"
**解决**:
```fortran
&INPUTPH
  lrpa = .false.
/
```

### 错误: "q-point not in grid"
**解决**:
```fortran
&INPUTPH
  ! 确保q点在定义的网格上
  ldisp = .true.
  nq1 = 2
  nq2 = 2
  nq3 = 2
/
```

## 6. 能带/DOS计算错误

### 错误: "bands: not enough bands"
**解决**:
```fortran
&SYSTEM
  nbnd = 20               ! 增加能带数
/
```

### 错误: "projwave: zero atomic wavefunctions"
**解决**:
```fortran
&SYSTEM
  ! 确保使用PAW或USPP赝势
  ! 检查赝势文件路径
/
```

## 7. GPU计算错误

### 错误: "CUDA out of memory"
**解决**:
```bash
# 增加GPU数量
mpirun -np 8 pw.x -in input.in  # 使用8个GPU

# 或降低截断能
# 减少k点
```

### 错误: "GPU-aware MPI not available"
**解决**:
```bash
# 编译时启用CUDA-aware MPI
cmake .. -DQE_ENABLE_MPI_GPU_AWARE=ON

# 或运行时禁用
export OMPI_MPI_CUDA_SUPPORT=0
```

## 8. 杂化泛函错误

### 错误: "forces for hybrid functionals + US/PAW not implemented"
**原因**: 杂化泛函不支持力计算
**解决**:
```fortran
&CONTROL
  calculation = 'scf'     ! 只做单点计算
  ! 结构优化需要先用GGA优化
/
```

### 错误: "EXX self-consistency not achieved"
**解决**:
```fortran
&ELECTRONS
  exx_maxstep = 100       ! 增加EXX步数
  conv_thr = 1.0D-7       ! 放宽收敛标准
/
```

## 9. 输入文件错误

### 错误: "card ATOMIC_POSITIONS not found"
**原因**: 输入文件格式错误
**解决**: 检查输入文件，确保所有必需部分存在

### 错误: "Error in routine read_pseudo"
**原因**: 赝势文件问题
**解决**:
```bash
# 检查赝势路径
export ESPRESSO_PSEUDO=/path/to/pseudo

# 检查文件是否存在
ls $ESPRESSO_PSEUDO/*.UPF
```

## 10. 性能优化

### 计算速度慢
```fortran
&SYSTEM
  ! 使用更粗的k点网格 (初步计算)
  ! 降低截断能
  
  ! k点并行
  kpar = 4
  
  ! 池化并行 (能带并行)
  npool = 2
/
```

### 磁盘空间不足
```fortran
&CONTROL
  wf_collect = .false.    ! 不收集波函数
  disk_io = 'low'         ! 减少磁盘I/O
/
```

## 调试技巧

### 1. 使用测试套件
```bash
cd build
make test
ctest -V -R pw_scf      # 详细输出
```

### 2. 逐步调试
```bash
# 先运行串行版本
pw.x -in input.in > output 2>&1

# 再测试并行
mpirun -np 2 pw.x -in input.in
```

### 3. 检查输入
```bash
# 使用输入验证工具
pw.x --check input.in
```

### 4. 监控输出
```bash
# 实时查看能量
tail -f pwscf.out | grep "!"

# 查看收敛情况
grep "estimated" pwscf.out
```

## 参考资源

- QE用户指南: https://www.quantum-espresso.org/Doc/user_guide/
- QE论坛: https://www.quantum-espresso.org/forum/
- GitLab Issues: https://gitlab.com/QEF/q-e/-/issues
