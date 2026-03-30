# 第一性原理计算方法详解

## 1. 结构优化 (Geometry Optimization)

### 原理
通过迭代调整原子位置和晶胞参数，找到体系能量最低的稳定结构。

### VASP实现

**关键INCAR参数**:
```
# 优化类型
ISIF = 2        # 2=离子弛豫, 3=离子+晶胞弛豫, 4=离子+晶胞形状
IBRION = 2      # 2=共轭梯度, 1=准牛顿法
NSW = 100       # 最大离子步数

# 收敛标准
EDIFFG = -0.01  # 力收敛标准 (eV/Å), 负值表示力
# 或 EDIFFG = 1E-4  # 能量收敛 (eV)

# 初始波函数和电荷密度
ISTART = 0      # 0=从头开始, 1=读取WAVECAR
ICHARG = 2      # 2=从头计算电荷密度

# 其他重要参数
POTIM = 0.5     # 时间步长 (IBRION=2时有效)
```

### QE实现

**pw.x输入**:
```fortran
&CONTROL
  calculation = 'relax'     ! 'relax'或'vc-relax'
  nstep = 100
  forc_conv_thr = 1.0D-3    ! 力收敛标准 (Ry/Bohr)
  etot_conv_thr = 1.0D-4    ! 能量收敛标准 (Ry)
/
&IONS
  ion_dynamics = 'bfgs'     ! 优化算法
/
&CELL
  cell_dynamics = 'bfgs'    ! 晶胞优化 (vc-relax时需要)
  press_conv_thr = 0.5      ! 压强收敛标准 (Kbar)
/
```

### 最佳实践

1. **分步优化**: 先固定晶胞优化原子位置，再优化晶胞
2. **初始结构**: 使用实验值或文献结构作为初始猜测
3. **收敛检查**: 确认所有力 < 收敛标准
4. **对称性**: 合理使用对称性加速计算

## 2. 能带计算 (Band Structure)

### 原理
计算电子在周期性势场中的能量-动量关系 E(k)。

### VASP实现

**步骤1: 自洽计算**
```
# INCAR
ISTART = 0
ICHARG = 2
ISMEAR = -5     # 四面体方法，用于能带计算

# KPOINTS (自动网格)
Automatic mesh
0
Gamma
11 11 11
0 0 0
```

**步骤2: 非自洽能带计算**
```
# INCAR
ISTART = 1      # 读取WAVECAR
ICHARG = 11     # 读取CHGCAR，非自洽
ISMEAR = 0      # Gaussian展宽
SIGMA = 0.05

# KPOINTS (高对称路径)
k-points along high symmetry line
40
Line-mode
Reciprocal
0.0 0.0 0.0    ! Gamma
0.5 0.0 0.5    ! X

0.5 0.0 0.5    ! X
0.5 0.25 0.75  ! W

0.5 0.25 0.75  ! W
0.0 0.0 0.0    ! Gamma
```

### QE实现

**步骤1: 自洽计算**
```fortran
&CONTROL
  calculation = 'scf'
  restart_mode = 'from_scratch'
/
&SYSTEM
  occupations = 'tetrahedra'  ! 用于能带计算
/
&ELECTRONS
  conv_thr = 1.0D-8
/
K_POINTS automatic
6 6 6 0 0 0
```

**步骤2: 能带计算**
```fortran
&CONTROL
  calculation = 'bands'
  restart_mode = 'restart'
/
&SYSTEM
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
  nbnd = 20       ! 增加能带数
/
K_POINTS crystal_b
4
0.000 0.000 0.000 40  ! Gamma, 40点
0.500 0.000 0.500 20  ! X, 20点
0.500 0.250 0.750 20  ! W, 20点
0.000 0.000 0.000 0   ! Gamma
```

### 高对称k点路径

常见晶体结构的高对称点:

**立方晶系 (FCC)**:
- Γ(0,0,0) → X(0.5,0,0.5) → W(0.5,0.25,0.75) → L(0.5,0.5,0.5) → Γ

**六角晶系**:
- Γ(0,0,0) → M(0.5,0,0) → K(1/3,1/3,0) → Γ → A(0,0,0.5)

## 3. 态密度计算 (Density of States)

### 总态密度 (TDOS)

**VASP**:
```
# INCAR (在自洽计算后)
ISMEAR = -5     # 四面体方法
LORBIT = 11     # 写入DOSCAR
NEDOS = 3001    # DOS点数
EMIN = -20      # 能量范围
EMAX = 20
```

**QE**:
```fortran
&CONTROL
  calculation = 'nscf'
  restart_mode = 'restart'
/
&SYSTEM
  occupations = 'tetrahedra'
  nbnd = 30
/
K_POINTS automatic
12 12 12 0 0 0  ! 更密的k点网格
```

然后使用`dos.x`计算DOS。

### 分波态密度 (PDOS)

**VASP**:
```
LORBIT = 11     # 11=轨道投影, 12=包括相位
NEDOS = 3001
```

**QE**:
```fortran
&PROJWFC
  outdir = './tmp'
  filpdos = 'system'
  filproj = 'system.proj'
  lsym = .true.
  lwrite_overlaps = .false.
  lbinary_data = .false.
/
```

## 4. 分子动力学 (Molecular Dynamics)

### 从头算分子动力学 (AIMD)

**VASP**:
```
# INCAR
IBRION = 0      # MD模拟
NSW = 1000      # 步数
POTIM = 1.0     # 时间步长 (fs)

# 系综选择
MDALGO = 2      # 2=Nose-Hoover, 1=Andersen, 3=Langevin
TEBEG = 300     # 起始温度
TEEND = 300     # 结束温度
SMASS = 0.5     # Nose-Hoover质量参数

# 或NVT系综
# TEBEG = 300
# SMASS = -1    # 速度标度

# 或微正则系综 (NVE)
# MDALGO = 0
```

**QE**:
```fortran
&CONTROL
  calculation = 'md'
  nstep = 1000
  dt = 20.0       ! 时间步长 (Ry原子单位, ~0.5 fs)
/
&IONS
  ion_dynamics = 'verlet'
  ion_temperature = 'nose'    ! 或'andersen', 'svr'
  tempw = 300.0               ! 目标温度 (K)
  fnosep = 40.0               ! Nose频率 (THz)
/
```

### 机器学习力场 (MLFF) - VASP 6.4+

```
# INCAR
ML_LMLFF = .TRUE.   ! 启用MLFF
ML_MODE = train     ! train=训练, run=预测, select=重选

# 训练参数
ML_CDOUB = 10       ! 训练数据倍增因子
ML_CTIFOR = 0.01    ! 力误差阈值

# 预测模式 (训练完成后)
# ML_MODE = run
# ML_FF = ML_FFN    ! 读取训练好的力场
```

## 5. 声子计算 (Phonon)

### 密度泛函微扰理论 (DFPT)

**QE** (推荐):
```fortran
# 步骤1: 自洽计算
calculation = 'scf'

# 步骤2: 声子计算 (ph.x输入)
&INPUTPH
  tr2_ph = 1.0D-14      ! 收敛阈值
  prefix = 'system'
  outdir = './tmp'
  fildyn = 'system.dyn'
  ldisp = .true.        ! 计算q点网格
  nq1 = 4, nq2 = 4, nq3 = 4  ! q点网格
  epsil = .true.        ! 计算介电常数
/
```

### 有限位移法

**VASP**:
```
# INCAR
IBRION = 6      ! 有限位移法
NFREE = 2       ! 位移数
POTIM = 0.015   ! 位移大小 (Å)
ISMEAR = 0
SIGMA = 0.1
```

## 6. 高级方法

### DFT+U (Hubbard U修正)

**VASP**:
```
LDAU = .TRUE.
LDAUTYPE = 2    ! 2=Dudarev方法
LDAUL = 2 -1    ! d轨道加U, p轨道不加
LDAUU = 4.0 0.0 ! U值 (eV)
LDAUJ = 0.0 0.0 ! J值 (eV)
LMAXMIX = 4     ! 混合电荷密度
```

**QE**:
```fortran
&SYSTEM
  lda_plus_u = .true.
  lda_plus_u_kind = 0     ! 0=Dudarev, 1=Liechtenstein
  Hubbard_U(1) = 4.0      ! 第一种类型的U值 (eV)
  Hubbard_U(2) = 0.0
/
```

### 杂化泛函 (HSE06)

**VASP**:
```
LHFCALC = .TRUE.
HFSCREEN = 0.2      ! HSE06
ALGO = Damped       ! 或 All, Normal
TIME = 0.4
PRECFOCK = Fast     ! 或 Normal, Accurate
NKRED = 2           ! k点缩减加速
```

**QE**:
```fortran
&SYSTEM
  input_dft = 'hse'
  exx_fraction = 0.25
  screening_parameter = 0.106
/
&ELECTRONS
  conv_thr = 1.0D-8
  mixing_mode = 'local-TF'
/
```

### GW近似

**VASP**:
```
ALGO = G0W0         ! 或 GW0, G0W0R, scGW0
LSPECTRAL = .TRUE.
NOMEGA = 50
OMEGAMAX = -1
```

**QE**:
使用`gw.x`或`perturbo`代码。

## 7. 收敛性测试

### 关键参数

1. **平面波截断能 (ENCUT/ecutwfc)**
   - 测试范围: 从默认值到1.5倍
   - 收敛标准: 能量变化 < 1 meV/atom

2. **k点网格**
   - 测试不同密度 (如 4×4×4, 6×6×6, 8×8×8)
   - 收敛标准: 能量变化 < 1 meV/atom

3. **真空层 (表面/分子)**
   - 测试 10Å, 15Å, 20Å
   - 确保能量收敛且没有周期性相互作用

### 自动化脚本

见 `scripts/convergence_test.py`
