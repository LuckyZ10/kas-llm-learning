# VASP输入文件示例

## 1. 基础自洽计算 (SCF)

### INCAR
```
# 基础自洽计算
SYSTEM = Si Bulk

# 电子步
ISTART = 0          # 从头开始
ICHARG = 2          # 从头计算电荷密度
ENCUT = 520         # 截断能 (eV)
EDIFF = 1E-6        # 电子步收敛标准
NELM = 100          # 最大电子步数

# 展宽
ISMEAR = 0          # Gaussian展宽
SIGMA = 0.05        # 展宽宽度 (eV)

# 并行
NCORE = 4           # 每核芯数
KPAR = 2            # k点并行组数
```

### POSCAR (硅晶体)
```
Si2
1.0
3.840000 0.000000 0.000000
1.920000 3.320083 0.000000
1.920000 1.106694 3.135510
Si
2
direct
0.000000 0.000000 0.000000
0.250000 0.250000 0.250000
```

### KPOINTS
```
Automatic mesh
0
Gamma
8 8 8
0 0 0
```

### POTCAR
```bash
# 使用PBE赝势
cat $POTCAR_PATH/PBE/Si/POTCAR > POTCAR
```

## 2. 结构优化

### INCAR
```
SYSTEM = Si Relaxation

# 初始设置
ISTART = 0
ICHARG = 2
ENCUT = 520

# 优化参数
IBRION = 2          # 共轭梯度
ISIF = 3            # 优化离子+晶胞
NSW = 100           # 最大离子步
EDIFFG = -0.01      # 力收敛标准 (eV/Å)
POTIM = 0.1         # 步长

# 电子步
EDIFF = 1E-6
NELM = 60
ISMEAR = 0
SIGMA = 0.05
```

## 3. 能带计算

### 步骤1: 自洽计算 INCAR
```
SYSTEM = Si SCF for Bands
ISTART = 0
ICHARG = 2
ENCUT = 520
ISMEAR = -5         # 四面体方法
EDIFF = 1E-8
```

### 步骤2: 能带计算 INCAR
```
SYSTEM = Si Band Structure
ISTART = 1          # 读取WAVECAR
ICHARG = 11         # 读取CHGCAR，非自洽
ENCUT = 520
ISMEAR = 0
SIGMA = 0.05
LORBIT = 11         # 投影态密度
NEDOS = 1001
```

### KPOINTS (能带路径)
```
k-points along high symmetry line
50
Line-mode
Reciprocal
0.000 0.000 0.000   ! Gamma
0.500 0.000 0.500   ! X

0.500 0.000 0.500   ! X
0.500 0.250 0.750   ! W

0.500 0.250 0.750   ! W
0.000 0.000 0.000   ! Gamma

0.000 0.000 0.000   ! Gamma
0.375 0.375 0.750   ! K

0.375 0.375 0.750   ! K
0.500 0.500 0.500   ! L
```

## 4. 态密度计算

### INCAR
```
SYSTEM = Si DOS
ISTART = 1
ICHARG = 11
ENCUT = 520
ISMEAR = -5
LORBIT = 11         # 投影到s,p,d轨道
NEDOS = 3001
EMIN = -15
EMAX = 15
```

## 5. 分子动力学 (NVT系综)

### INCAR
```
SYSTEM = Si MD NVT

# MD设置
IBRION = 0          # MD模拟
NSW = 1000          # 步数
POTIM = 1.0         # 时间步长 (fs)

# 温度控制
MDALGO = 2          # Nose-Hoover
TEBEG = 300         # 起始温度 (K)
TEEND = 300         # 目标温度 (K)
SMASS = 0.5         # Nose-Hoover质量

# 电子设置
ENCUT = 520
ISMEAR = 0
SIGMA = 0.1
EDIFF = 1E-5
NELM = 50

# 输出
NBLOCK = 1          # 每步都写入
KBLOCK = 50         # XDATCAR写入频率
```

## 6. 机器学习力场训练 (VASP 6.4+)

### INCAR
```
SYSTEM = Si MLFF Training

# MLFF设置
ML_LMLFF = .TRUE.
ML_MODE = train     # 训练模式
ML_CDOUB = 10       # 数据倍增
ML_CTIFOR = 0.01    # 力误差阈值

# MD设置 (用于生成训练数据)
IBRION = 0
NSW = 5000
POTIM = 1.0
MDALGO = 2
TEBEG = 300
TEEND = 1200        # 升温以探索构型空间
SMASS = 0.5

# 电子设置
ENCUT = 520
ISMEAR = 0
SIGMA = 0.1
EDIFF = 1E-5
```

### MLFF预测模式 INCAR
```
SYSTEM = Si MLFF Production

# MLFF预测
ML_LMLFF = .TRUE.
ML_MODE = run       # 纯预测模式
ML_FF = ML_FFN      # 读取训练好的力场

# MD设置
IBRION = 0
NSW = 100000        # 长MD轨迹
POTIM = 1.0
MDALGO = 2
TEBEG = 300
TEEND = 300
SMASS = 0.5

# 注意: 预测模式下不需要高精度电子设置
ENCUT = 400         # 可降低截断能
ISMEAR = 0
SIGMA = 0.1
```

## 7. DFT+U计算

### INCAR
```
SYSTEM = NiO DFT+U

# 基础设置
ENCUT = 520
ISMEAR = -5

# DFT+U参数
LDAU = .TRUE.
LDAUTYPE = 2        # Dudarev方法
LDAUL = 2 -1        # Ni-d加U, O不加
LDAUU = 6.0 0.0     # U值 (eV)
LDAUJ = 0.0 0.0     # J值 (eV)
LMAXMIX = 4         # 电荷密度混合

# 磁设置
ISPIN = 2           # 自旋极化
MAGMOM = 2.0 -2.0 0 0  # Ni1↑ Ni2↓ O无磁矩
```

## 8. HSE06杂化泛函

### INCAR
```
SYSTEM = Si HSE06

# 杂化泛函
LHFCALC = .TRUE.
HFSCREEN = 0.2      # HSE06
ALGO = Damped
TIME = 0.4
PRECFOCK = Fast
NKRED = 2           # k点缩减

# 标准参数
ENCUT = 520
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-6

# 能带计算时
# ICHARG = 11
# ISMEAR = 0
```

## 9. 声子计算 (有限位移法)

### INCAR
```
SYSTEM = Si Phonon

# 有限位移法
IBRION = 6          # 有限位移
NFREE = 2           # 正负位移
POTIM = 0.015       # 位移大小 (Å)

# 高精度
ENCUT = 520
EDIFF = 1E-8
ISMEAR = 0
SIGMA = 0.05

# 输出
LEPSILON = .TRUE.   # 计算介电常数
LRPA = .FALSE.
```

## 10. 表面计算 (Slab模型)

### INCAR
```
SYSTEM = Si Surface

# 基础设置
ENCUT = 520
ISMEAR = 0
SIGMA = 0.05

# 偶极校正 (表面)
IDIPOL = 3          # z方向偶极校正
LDIPOL = .TRUE.
LVHAR = .TRUE.

# 优化
IBRION = 2
ISIF = 2            # 只优化离子
NSW = 100
EDIFFG = -0.01
```

### POSCAR (Si(001)表面)
```
Si surface
1.0
3.840000 0.000000 0.000000
0.000000 3.840000 0.000000
0.000000 0.000000 30.00000   # 大c轴，包含真空层
Si
8
direct
0.000000 0.000000 0.100000
0.500000 0.000000 0.100000
0.000000 0.500000 0.100000
0.500000 0.500000 0.100000
0.250000 0.250000 0.150000
0.750000 0.250000 0.150000
0.250000 0.750000 0.150000
0.750000 0.750000 0.150000
```

## 11. 过渡态搜索 (NEB)

### INCAR
```
SYSTEM = Reaction NEB

# NEB设置
IBRION = 3          # MD方式NEB
IOPT = 1            # 优化算法 (1=QuickMin)
ICHAIN = 0          # NEB
SPRING = -5         # 弹簧常数
LCLIMB = .TRUE.     # 攀爬图像

# 图像数 (在POTCAR中指定)
IMAGES = 5          # 中间图像数

# 标准参数
ENCUT = 520
EDIFFG = -0.05      # NEB收敛较松
NSW = 500
ISMEAR = 0
SIGMA = 0.05
```

## 12. 光学性质计算

### INCAR
```
SYSTEM = Si Optics

# 光学计算
LOPTICS = .TRUE.    # 计算光学矩阵元
CSHIFT = 0.1        # 半经验展宽
NEDOS = 2000

# 自洽设置
ISTART = 0
ICHARG = 2
ENCUT = 520
ISMEAR = -5

# 或GW/BSE计算
# ALGO = G0W0
# LSPECTRAL = .TRUE.
```
