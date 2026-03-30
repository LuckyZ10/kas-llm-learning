# Quantum ESPRESSO 常见错误排查指南

## 1. 编译错误

### 错误: "configure: error: Fortran compiler cannot create executables"
**原因**: Fortran编译器未安装或配置错误
**解决**:
```bash
# 安装Fortran编译器
sudo apt-get install gfortran  # Ubuntu/Debian
sudo yum install gcc-gfortran   # CentOS/RHEL

# 验证
which gfortran
gfortran --version

# 配置时指定
./configure FC=gfortran F77=gfortran
```

### 错误: "BLAS/LAPACK not found"
**原因**: 线性代数库缺失
**解决**:
```bash
# 安装OpenBLAS
sudo apt-get install libopenblas-dev liblapack-dev

# 或指定Intel MKL路径
./configure BLAS_LIBS="-L/opt/intel/mkl/lib/intel64 -lmkl_rt" \
            LAPACK_LIBS="-L/opt/intel/mkl/lib/intel64 -lmkl_rt"

# 或使用内部库
./configure --with-internal-blas --with-internal-lapack
```

### 错误: "FFTW3 not found"
**解决**:
```bash
# Ubuntu
sudo apt-get install libfftw3-dev

# 或从源码安装
wget http://www.fftw.org/fftw-3.3.10.tar.gz
tar -xzf fftw-3.3.10.tar.gz
cd fftw-3.3.10
./configure --enable-mpi --enable-openmp
make && sudo make install

# QE配置
./configure FFTW_INCLUDE="/usr/local/include" \
            FFTW_LIBS="-L/usr/local/lib -lfftw3"
```

### 错误: "MPI not found" 或 "mpif90 not found"
**解决**:
```bash
# 安装OpenMPI
sudo apt-get install openmpi-bin libopenmpi-dev

# 配置
./configure MPIF90=mpif90

# 或禁用MPI (串行编译)
./configure --disable-parallel
```

### 错误: "configure: WARNING: ELPA not found"
**说明**: 这是警告，不影响基本功能
**解决** (如需高性能对角化):
```bash
# 下载并编译ELPA
# 然后在QE配置中指定
./configure ELPA_LIBS="-L/path/to/elpa/lib -lelpa" \
            ELPA_INCLUDE="-I/path/to/elpa/include"
```

---

## 2. 输入文件错误

### 错误: "Error in routine input (1): unknown card in input"
**原因**: 输入文件中有不识别的关键词
**解决**:
```bash
# 1. 检查拼写错误
# 2. 确认输入顺序 (namelist → cards)
# 3. 检查是否有隐藏字符

# 查看支持的输入
grep -r "&SYSTEM" /path/to/q-e/Modules/
```

### 错误: "reading input namelist system"
**原因**: namelist格式错误
**解决**:
```bash
# 常见问题:
# 1. namelist未正确结束
&SYSTEM
  ...
/              # 必须有结束符/

# 2. 参数间缺少逗号
ecutwfc = 50,   # 必须有逗号
ecutrho = 400,

# 3. 字符串未加引号
calculation = 'scf'  # 正确
calculation = scf    # 错误
```

### 错误: "ibrav = 0: celldm(1) must be different from 0"
**原因**: ibrav=0时需要指定晶格向量
**解决**:
```bash
# 错误
&SYSTEM
ibrav = 0
celldm(1) = 10.0  # 忽略，使用CELL_PARAMETERS
/

# 正确
&SYSTEM
ibrav = 0
/
CELL_PARAMETERS angstrom
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
```

### 错误: "K-point list is empty or not found"
**解决**:
```bash
# 自动网格
K_POINTS automatic
4 4 4 0 0 0

# Γ点
K_POINTS gamma

# 显式列表
K_POINTS crystal
4
0.0 0.0 0.0 1.0
0.5 0.0 0.0 1.0
0.0 0.5 0.0 1.0
0.0 0.0 0.5 1.0
```

### 错误: "pseudo potential file not found"
**解决**:
```bash
# 1. 检查文件名拼写
# 2. 设置环境变量或伪势路径
export ESPRESSO_PSEUDO="/path/to/pseudo"

# 或在输入中指定
pseudo_dir = '/path/to/pseudo/',

# 3. 下载伪势库
# 从 https://www.quantum-espresso.org/pseudopotentials/
```

---

## 3. SCF不收敛

### 症状: "convergence NOT achieved"
**原因与解决**:

```bash
# 方法1: 增加混合beta值
&ELECTRONS
  mixing_beta = 0.7    # 默认0.7，增加到0.8-0.9
/

# 方法2: 增加电子步数
  electron_maxstep = 200  # 默认100

# 方法3: 改变混合模式
  mixing_mode = 'local-TF'  # 或 'TF'

# 方法4: 降低展宽 (金属体系)
  degauss = 0.01  # 从0.02降低

# 方法5: 开启smearing
  occupations = 'smearing',
  smearing = 'mv',    # Methfessel-Paxton
  degauss = 0.01,
```

### 症状: 能量震荡
**解决**:
```bash
&ELECTRONS
  mixing_beta = 0.4      # 降低混合参数
  mixing_ndim = 12       # 增加Pulay历史
  diagonalization = 'david'  # 更稳定
/

# 或分步计算
# 1. 先用低截断能收敛
ecutwfc = 30
# 2. 用前一步波函数重启
startingwfc = 'file'
ecutwfc = 60
```

### 症状: 磁性体系不收敛
**解决**:
```bash
&SYSTEM
  nspin = 2,
  starting_magnetization(1) = 0.5,  # 初始磁矩
/
&ELECTRONS
  mixing_beta = 0.3,
  mixing_spin_mag = 0.4,
  mixing_angle_mag = 0.4,
/
```

---

## 4. 结构优化问题

### 错误: "bfgs converged to a wrong solution"
**原因**: BFGS陷入鞍点或初始结构不合理
**解决**:
```bash
# 方法1: 使用阻尼动力学
&IONS
  ion_dynamics = 'damp',
  pot_extrapolation = 'second_order',
  wfc_extrapolation = 'second_order',
/

# 方法2: 放宽收敛标准先弛豫
forc_conv_thr = 1.0d-3  # 先放宽
# 再收紧到1.0d-4
```

### 错误: "press is not the target pressure"
**解决** (变胞优化):
```bash
&CONTROL
  calculation = 'vc-relax',
  press = 0.0,           # 目标压力 (kBar)
  press_conv_thr = 0.5,  # 压力收敛标准
/
&IONS
  ion_dynamics = 'bfgs',
/
&CELL
  cell_dynamics = 'bfgs',
  cell_dofree = 'all',   # 或 'xyz', 'shape'
/
```

### 警告: "Proceeding with random crystal"
**原因**: 原子位置过于接近或输入错误
**解决**:
```bash
# 检查POSCAR
# 检查晶格常数 (angstrom vs bohr)
# 检查原子坐标单位
ATOMIC_POSITIONS angstrom  # 或 bohr, crystal
```

---

## 5. 声子计算错误

### 错误: "q not in the IBZ"
**原因**: 请求的q点不在不可约布里渊区内
**解决**:
```bash
# 使用QE生成q点列表
# ph.x会自动处理对称性

# 或显式指定irreducible q点
&INPUTPH
  ldisp = .true.,
  nq1 = 4, nq2 = 4, nq3 = 4,  # q网格
/
```

### 错误: "error in star_q"
**原因**: 低对称性q点导致星群计算失败
**解决**:
```bash
# 添加小位移打破对称性
&SYSTEM
  nosym = .true.,    # 临时关闭对称性
  noinv = .true.,
/

# 或在ph.x中设置
&INPUTPH
  reduce_io = .true.,
/
```

### 错误: "wrong representation of q point"
**原因**: 声子模式对称性分析失败
**解决**:
```bash
# 跳过模式分析
&INPUTPH
  lraman = .false.,
  elop = .false.,
/

# 或使用不同q点
```

### 错误: "The dynamical matrix was not symmetrized"
**解决**:
```bash
# 在q2r.x后使用
&INPUT
  loto_2d = .true.,  # 如果是2D体系
  nosym = .false.,
/

# 确保k点网格足够密
```

---

## 6. 并行计算问题

### 错误: "mpirun detected that one or more processes exited with non-zero status"
**解决**:
```bash
# 1. 检查内存
ulimit -s unlimited
ulimit -v unlimited

# 2. 减少并行度或npool
mpirun -np 32 pw.x -npool 4 ...  # 从8减少到4

# 3. 检查磁盘空间
df -h $TMPDIR

# 4. 使用更大的outdir
outdir = '/scratch/large_disk/',
```

### 错误: "bad total algorithmic parallelism"
**原因**: 任务分配不均
**解决**:
```bash
# 优化npool设置
# 原则: k点总数应被npool整除

# 例如: 64个k点
mpirun -np 64 pw.x -npool 8 ...  # 8池 × 每池8核
mpirun -np 64 pw.x -npool 16 ... # 16池 × 每池4核

# 检查k点分布
# 在输出中查看 "K-points division"
```

### 错误: "cannot reshape array"
**原因**: 内存不足
**解决**:
```bash
# 1. 减少波函数缓存
disk_io = 'low',  # 或 'none' (需要更多内存)

# 2. 减少进程数，增加每进程内存

# 3. 使用更小截断能测试
ecutwfc = 40  # 暂时降低
```

---

## 7. Wannier90接口问题

### 错误: "wannier90.x not found"
**解决**:
```bash
# 1. 编译时启用Wannier90
./configure --with-wannier

# 2. 或指定路径
./configure WANNIER90_LIBS="-L/path/to/wannier90 -lwannier" \
            WAN90_SHARED="path/to/wannier90/wannier90.x"
```

### 错误: "Error: reading wannier90.win"
**原因**: Wannier90输入文件错误
**解决**:
```bash
# 常见错误:
# 1. mp_grid与k点不匹配
mp_grid : 4 4 4  # 必须与pw.x nscf k点一致

# 2. 投影未定义或错误
begin projections
Fe:l=0;l=2
As:l=1
end projections

# 3. num_bands与num_wann不匹配
num_bands = 20
num_wann = 14
```

### 错误: "disentanglement not converged"
**解决**:
```bash
# 调整外层能量窗口
dis_win_min = -10.0
dis_win_max = 15.0

# 增加迭代次数
dis_num_iter = 2000

# 冻结内层能带
dis_froz_min = -8.0
dis_froz_max = 5.0
```

---

## 8. 后处理工具错误

### 错误: "charge density is incomplete"
**解决**:
```bash
# 重新计算电荷密度
calculation = 'scf',
# 确保
lda_plus_u = .true.,  # 如果使用DFT+U
# 且Hubbard_U正确设置

# 或使用pp.x重新生成
cat > pp.in << 'EOF'
&INPUTPP
  prefix = 'mycalc',
  outdir = './tmp',
  plot_num = 0,    # 电荷密度
  filplot = 'rho.dat',
/
&PLOT
  iflag = 3,       # 3D输出
  output_format = 6,  # XCrySDen
  fileout = 'rho.xsf',
/
EOF
pp.x < pp.in
```

### 错误: "bands.x: no k-points found"
**解决**:
```bash
# pw.x nscf计算时必须指定
&SYSTEM
  nbnd = 20,       # 空带
/
K_POINTS crystal_b
4
0.0 0.0 0.0 20
0.5 0.0 0.0 20
0.5 0.5 0.0 20
0.0 0.0 0.0 1
```

---

## 9. GPU计算问题

### 错误: "CUDA out of memory"
**解决**:
```bash
# 1. 减少batch大小或使用更少GPU
# 2. 使用混合精度
./configure --enable-cuda --with-cuda-cc=80

# 3. 在输入中控制
&SYSTEM
  tqr = .false.,   # 使用标准算法
/
```

### 错误: "GPU not detected"
**解决**:
```bash
# 检查CUDA
nvidia-smi
nvcc --version

# 配置时指定CUDA路径
./configure --with-cuda=/usr/local/cuda-11.8 \
            --with-cuda-cc=80

# 运行时指定GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## 10. 性能优化建议

```bash
# 1. k点并行优化
# 总核数 = npool × nk_per_pool × nbgrp
# 最优: npool ≈ k点总数

# 2. 大体系优化
&SYSTEM
  tbeta_smoothing = .true.,
  tq_smoothing = .true.,
  tprnfor = .false.,  # 如不需要力
  tstress = .false.,  # 如不需要应力
/

# 3. 磁盘IO优化
outdir = '/fast/ssd/$USER/',
disk_io = 'medium',

# 4. 混合并行 (OpenMP + MPI)
export OMP_NUM_THREADS=4
mpirun -np 16 pw.x ...
```

---

## 参考资源

- QE用户指南: https://www.quantum-espresso.org/documentation/
- QE论坛: https://lists.quantum-espresso.org/mailman/listinfo
- GitHub Issues: https://github.com/QEF/q-e/issues
- 教程: https://github.com/QEF/qe-tutorials

---

*文档版本: 1.0*
*最后更新: 2026-03-08*
