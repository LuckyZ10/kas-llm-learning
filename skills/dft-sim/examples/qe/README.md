# Quantum ESPRESSO输入文件示例

## 1. 基础自洽计算 (SCF)

### pw.x输入 (硅晶体)
```fortran
&CONTROL
  calculation = 'scf'
  restart_mode = 'from_scratch'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  tprnfor = .true.
  tstress = .true.
/
&SYSTEM
  ibrav = 2           # FCC晶格
  celldm(1) = 10.26   # 晶格常数 (a.u.)
  nat = 2
  ntyp = 1
  ecutwfc = 40        # 波函数截断能 (Ry)
  ecutrho = 320       # 电荷密度截断能 (Ry)
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01      # 展宽 (Ry)
/
&ELECTRONS
  conv_thr = 1.0D-8
  mixing_beta = 0.7
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25

K_POINTS automatic
6 6 6 0 0 0
```

## 2. 结构优化

### 离子弛豫 (relax)
```fortran
&CONTROL
  calculation = 'relax'
  restart_mode = 'from_scratch'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  tprnfor = .true.
  tstress = .true.
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0D-8
  mixing_beta = 0.7
/
&IONS
  ion_dynamics = 'bfgs'
  ion_positions = 'default'
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.26 0.26 0.26   # 稍微偏离平衡位置

K_POINTS automatic
6 6 6 0 0 0
```

### 离子+晶胞弛豫 (vc-relax)
```fortran
&CONTROL
  calculation = 'vc-relax'
  restart_mode = 'from_scratch'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.5    # 初始猜测
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0D-8
/
&IONS
  ion_dynamics = 'bfgs'
/
&CELL
  cell_dynamics = 'bfgs'
  press = 0.0         # 目标压强 (Kbar)
  press_conv_thr = 0.5
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25

K_POINTS automatic
6 6 6 0 0 0
```

## 3. 能带计算

### 步骤1: 自洽计算
```fortran
&CONTROL
  calculation = 'scf'
  restart_mode = 'from_scratch'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  occupations = 'tetrahedra'  # 用于能带计算
/
&ELECTRONS
  conv_thr = 1.0D-8
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25

K_POINTS automatic
8 8 8 0 0 0
```

### 步骤2: 能带计算 (bands.x)
```fortran
&CONTROL
  calculation = 'bands'
  restart_mode = 'restart'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  nbnd = 20           # 增加能带数
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0D-8
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25

K_POINTS crystal_b
6
0.000 0.000 0.000 40   # Gamma
0.500 0.000 0.500 20   # X
0.500 0.250 0.750 20   # W
0.000 0.000 0.000 20   # Gamma
0.375 0.375 0.750 20   # K
0.500 0.500 0.500 0    # L
```

### 后处理: bands.x输入
```fortran
&BANDS
  prefix = 'si'
  outdir = './tmp'
  filband = 'si.bands'
  lsym = .true.       # 识别高对称点
/
```

## 4. 态密度计算

### 步骤1: 非自洽计算 (nscf)
```fortran
&CONTROL
  calculation = 'nscf'
  restart_mode = 'restart'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  nbnd = 20
  occupations = 'tetrahedra'
/
&ELECTRONS
  conv_thr = 1.0D-8
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25

K_POINTS automatic
12 12 12 0 0 0      # 更密的k点网格
```

### 步骤2: DOS计算 (dos.x)
```fortran
&DOS
  prefix = 'si'
  outdir = './tmp'
  fildos = 'si.dos'
  degauss = 0.02      # 展宽 (Ry)
  Emin = -10.0        # 能量范围 (eV)
  Emax = 10.0
  DeltaE = 0.01       # 能量步长
/
```

### 投影态密度 (projwfc.x)
```fortran
&PROJWFC
  prefix = 'si'
  outdir = './tmp'
  filpdos = 'si.pdos'
  filproj = 'si.proj'
  lsym = .true.
  lwrite_overlaps = .false.
  lbinary_data = .false.
  degauss = 0.02
  DeltaE = 0.01
/
```

## 5. 分子动力学

### NVT系综 (Nose-Hoover)
```fortran
&CONTROL
  calculation = 'md'
  restart_mode = 'from_scratch'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  nstep = 1000
  dt = 20.0           # 时间步长 (Ry a.u., ~0.48 fs)
  tprnfor = .true.
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0D-6
  electron_dynamics = 'damp'
  electron_damping = 0.2
/
&IONS
  ion_dynamics = 'verlet'
  ion_temperature = 'nose'
  tempw = 300.0       # 目标温度 (K)
  fnosep = 40.0       # Nose频率 (THz)
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25

K_POINTS automatic
4 4 4 0 0 0
```

### NVE系综 (微正则)
```fortran
&CONTROL
  calculation = 'md'
  restart_mode = 'from_scratch'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  nstep = 1000
  dt = 20.0
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
/
&ELECTRONS
  conv_thr = 1.0D-6
/
&IONS
  ion_dynamics = 'verlet'
  ion_temperature = 'not_controlled'  # NVE
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25

K_POINTS automatic
4 4 4 0 0 0
```

## 6. 声子计算 (DFPT)

### 步骤1: 自洽计算
```fortran
&CONTROL
  calculation = 'scf'
  restart_mode = 'from_scratch'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
/
&ELECTRONS
  conv_thr = 1.0D-12  # 更高精度
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25

K_POINTS automatic
8 8 8 0 0 0
```

### 步骤2: 声子计算 (ph.x)
```fortran
&INPUTPH
  tr2_ph = 1.0D-14    # 收敛阈值
  prefix = 'si'
  outdir = './tmp'
  fildyn = 'si.dyn'
  ldisp = .true.      # 计算q点网格
  nq1 = 4             # q点网格
  nq2 = 4
  nq3 = 4
  epsil = .true.      # 计算介电常数
  trans = .true.      # 计算声子
/
```

### 步骤3: 后处理

**q2r.x** (实空间力常数):
```fortran
&INPUT
  fildyn = 'si.dyn'
  flfrc = 'si.fc'
  zasr = 'simple'     # 声学求和规则
/
```

**matdyn.x** (声子色散):
```fortran
&INPUT
  flfrc = 'si.fc'
  flfrq = 'si.freq'
  asr = 'simple'
  q_in_band_form = .true.
/
6                                   # q点路径点数
0.000 0.000 0.000 40                # Gamma
0.500 0.000 0.500 20                # X
0.500 0.250 0.750 20                # W
0.000 0.000 0.000 20                # Gamma
0.375 0.375 0.750 20                # K
0.500 0.500 0.500 0                 # L
```

## 7. DFT+U计算

```fortran
&CONTROL
  calculation = 'scf'
  restart_mode = 'from_scratch'
  prefix = 'nio'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 7.88
  nat = 2
  ntyp = 2
  ecutwfc = 60
  ecutrho = 480
  nspin = 2           # 自旋极化
  lda_plus_u = .true.
  lda_plus_u_kind = 0 # 0=Dudarev, 1=Liechtenstein
  Hubbard_U(1) = 6.0  # Ni的U值 (eV)
  Hubbard_U(2) = 0.0  # O的U值
  starting_magnetization(1) = 0.5   # Ni初始磁矩
/
&ELECTRONS
  conv_thr = 1.0D-8
  mixing_beta = 0.3   # DFT+U需要较小的mixing
/
ATOMIC_SPECIES
 Ni 58.693 Ni.pbe-spn-kjpaw_psl.1.0.0.UPF
 O  15.999 O.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
 Ni 0.00 0.00 0.00
 O  0.50 0.50 0.50

K_POINTS automatic
6 6 6 0 0 0
```

## 8. HSE06杂化泛函

```fortran
&CONTROL
  calculation = 'scf'
  restart_mode = 'from_scratch'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  input_dft = 'hse'   # HSE泛函
  exx_fraction = 0.25
  screening_parameter = 0.106
  nqx1 = 4            # EXX q点网格
  nqx2 = 4
  nqx3 = 4
/
&ELECTRONS
  conv_thr = 1.0D-8
  mixing_mode = 'local-TF'
  diagonalization = 'david'
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25

K_POINTS automatic
4 4 4 0 0 0         # HSE需要较小的k点网格
```

## 9. 表面计算 (Slab模型)

```fortran
&CONTROL
  calculation = 'relax'
  restart_mode = 'from_scratch'
  prefix = 'si_surf'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  tprnfor = .true.
/
&SYSTEM
  ibrav = 0           # 自由晶格
  nat = 8
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
  assume_isolated = 'esm'     # 偶极校正
  esm_bc = 'bc2'              # 表面边界条件
  esm_w = 20.0                # 真空层厚度 (a.u.)
/
&ELECTRONS
  conv_thr = 1.0D-8
/
&IONS
  ion_dynamics = 'bfgs'
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF

CELL_PARAMETERS angstrom
3.840000 0.000000 0.000000
0.000000 3.840000 0.000000
0.000000 0.000000 30.00000   # 大z方向，包含真空层

ATOMIC_POSITIONS angstrom
Si 0.000000 0.000000 5.000000
Si 1.920000 1.920000 5.000000
Si 0.000000 1.920000 6.357500
Si 1.920000 0.000000 6.357500
Si 0.960000 0.960000 7.715000
Si 2.880000 2.880000 7.715000
Si 0.960000 2.880000 9.072500
Si 2.880000 0.960000 9.072500

K_POINTS automatic
6 6 1 0 0 0         # z方向k点=1
```

## 10. 能带结构 + 投影分析

```fortran
# 先完成scf和bands计算，然后:

# 1. projwfc.x输入
&PROJWFC
  prefix = 'si'
  outdir = './tmp'
  filpdos = 'si.pdos'
  filproj = 'si.proj'
  lsym = .true.
  lorbit = 11
/

# 2. bands.x输入 (带权重)
&BANDS
  prefix = 'si'
  outdir = './tmp'
  filband = 'si.bands'
  lsym = .true.
  spin_component = 1
/

# 3. plotband.x输入 (交互式)
# 命令行: plotband.x
# 输入:
# si.bands
# -10 10        # 能量范围
# si.plotband.ps
# 0.0
# 0.0 -10 10    # 绘图范围
```
