# 磁性计算方法：SOC、非共线磁性与磁各向异性能

## 简介

磁性材料的第一性原理计算涉及多个层次：从简单的共线磁性到复杂的非共线磁性，再到考虑自旋-轨道耦合(SOC)的相对论效应。磁各向异性能(MAE)是磁性材料的关键性质，决定了磁矩的优选方向。

---

## 理论基础

### 自旋-轨道耦合 (Spin-Orbit Coupling)

SOC是相对论效应，描述电子自旋与轨道角动量的耦合：

$$H_{SOC} \propto \mathbf{\sigma} \cdot \mathbf{L}$$

其中 $\mathbf{\sigma}$ 是Pauli自旋算符，$\mathbf{L}$ 是轨道角动量算符。

**物理效应**:
- 磁晶各向异性 (Magnetocrystalline Anisotropy)
- Dzyaloshinskii-Moriya相互作用 (DMI)
- 能带劈裂 (Band splitting)
- 拓扑绝缘体态

### 磁各向异性能 (MAE)

MAE定义为磁矩沿不同方向时的能量差：

$$E_{MAE} = E_{hard} - E_{easy}$$

或常用定义：

$$E_{MAE} = E_{\parallel}^{SOC} - E_{\perp}^{SOC}$$

其中：
- $E_{\parallel}$: 磁矩平行于某个晶面/轴的能量
- $E_{\perp}$: 磁矩垂直于该晶面/轴的能量

**MAE符号含义**:
- MAE > 0: 易轴垂直于平面 (out-of-plane)
- MAE < 0: 易轴在平面内 (in-plane)

---

## VASP实现

### 基础设置

**使用vasp_ncl**: SOC计算必须使用非共线版本VASP

```bash
mpirun -np 16 vasp_ncl
```

### 共线磁性计算 (Collinear)

作为MAE计算的第一步：

```
# INCAR - 共线计算
ISPIN = 2
MAGMOM = 5 5 -5 -5    # 每个离子的初始磁矩 (沿z轴)
PREC = Accurate
LREAL = .FALSE.
EDIFF = 1E-8          # 高精度收敛
ISMEAR = 0
SIGMA = 0.05
LWAVE = .TRUE.
LCHARG = .TRUE.
```

### 非共线磁性计算 (Noncollinear)

**无SOC的非共线计算**:

```
# INCAR
LNONCOLLINEAR = .TRUE.
MAGMOM = 0 0 3  0 0 -3  0 0 3  0 0 -3   # Mx My Mz for each ion
NBANDS = [2 * 共线计算的NBANDS]          # 能带数翻倍
PREC = Accurate
LREAL = .FALSE.
ISYM = -1                 # 建议关闭对称性
```

### 自旋-轨道耦合计算 (SOC)

**完整SOC设置**:

```
# INCAR
LSORBIT = .TRUE.          # 自动设置 LNONCOLLINEAR = .TRUE.
MAGMOM = 0 0 3  0 0 -3  0 0 3  0 0 -3
SAXIS = 0 0 1             # 自旋量子化轴 (默认z轴)
NBANDS = [2 * 共线NBANDS]

# 高精度设置
PREC = Accurate
LREAL = .FALSE.
EDIFF = 1E-8
ISYM = -1                 # 关闭对称性
GGA_COMPAT = .FALSE.      # 提高GGA数值精度

# 混合参数
AMIX = 0.1
BMIX = 0.00001
AMIX_MAG = 0.2
BMIX_MAG = 0.00001

# DFT+U (如需要)
LMAXMIX = 4               # d元素设为4, f元素设为6
```

**关键参数说明**:

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `LSORBIT` | 开启SOC | .TRUE. |
| `SAXIS` | 自旋量子化轴 | 0 0 1 (z轴) |
| `MAGMOM` | 初始磁矩 (Mx My Mz) | 每个离子3个值 |
| `NBANDS` | 能带数 | 共线计算的2倍 |
| `ISYM` | 对称性 | -1 (关闭) |
| `GGA_COMPAT` | GGA兼容性 | .FALSE. |
| `LMAXMIX` | 最大角动量混合 | 4 (d), 6 (f) |

### 磁各向异性能 (MAE) 计算

#### 方法一：非自洽计算 (推荐，快速)

**Step 1**: 高精度共线计算获取电荷密度

```
# INCAR - collinear
ISPIN = 2
MAGMOM = 5 5 -5 -5
PREC = Accurate
LREAL = .FALSE.
EDIFF = 1E-9            # 极高精度
ISMEAR = 0
SIGMA = 0.05
LWAVE = .TRUE.
LCHARG = .TRUE.
```

**Step 2**: 非自洽SOC计算不同方向

```
# INCAR - SOC non-selfconsistent
LSORBIT = .TRUE.
ICHARG = 11             # 从CHGCAR读取电荷密度
MAGMOM = 0 0 5          # 磁矩沿z方向
SAXIS = 0 0 1           # [001] 方向
NBANDS = [2 * 共线NBANDS]
PREC = Accurate
LREAL = .FALSE.
EDIFF = 1E-9
ISYM = -1
```

改变 `SAXIS` 计算不同方向：
- `SAXIS = 0 0 1`: [001] 方向 (垂直)
- `SAXIS = 1 0 0`: [100] 方向 (平面x)
- `SAXIS = 0 1 0`: [010] 方向 (平面y)
- `SAXIS = 1 1 0`: [110] 方向
- `SAXIS = 1 1 1`: [111] 方向

**MAE计算**:

```python
# 从OUTCAR提取能量
E_001 = extract_energy('001/OUTCAR')  # 垂直方向
E_100 = extract_energy('100/OUTCAR')  # 平面方向

MAE = E_100 - E_001  # meV/unit cell

# 各向异性常数 K_i (J/m^3)
K_i = MAE * 1.602e-22 / volume  # volume in Ang^3
```

#### 方法二：自洽计算 (精确但耗时)

对每个方向进行完全自洽计算：

```
# INCAR - SOC selfconsistent
LSORBIT = .TRUE.
MAGMOM = 0 0 5
SAXIS = 0 0 1
NBANDS = [2 * 共线NBANDS]
PREC = Accurate
LREAL = .FALSE.
EDIFF = 1E-8
ISYM = -1
GGA_COMPAT = .FALSE.

# 混合参数 (重要!)
AMIX = 0.1
BMIX = 0.00001
AMIX_MAG = 0.2
BMIX_MAG = 0.00001
```

### 磁矩约束计算

使用约束磁矩方法研究特定自旋构型：

```
# INCAR
LSORBIT = .TRUE.
I_CONSTRAINED_M = 1     # 开启磁矩约束
LAMBDA = 10             # 约束强度 (惩罚函数系数)
M_CONSTR = 0 0 3        # 目标磁矩方向 (每个离子)
MAGMOM = 0 0 3
SAXIS = 0 0 1
```

**应用**: 计算磁交换相互作用、DMI、四态法(four-state mapping)

### 轨道磁矩计算

```
# INCAR
LSORBIT = .TRUE.
LORBMOM = .TRUE.        # 输出轨道磁矩
LORBIT = 11             # 输出详细磁矩信息
```

OUTCAR中查找：
```
magnetization (x)   magnetization (y)   magnetization (z)
# of ion     s       p       d       tot   ->   s       p       d       tot
--------------------------------------------------------------------------------
    1       0.000   0.000   2.123   2.123      0.000   0.000   0.234   0.234
```

---

## Quantum ESPRESSO实现

### 基础SOC计算

**输入文件关键设置**:

```fortran
&SYSTEM
  noncolin = .true.       ! 非共线计算
  lspinorb = .true.       ! 自旋轨道耦合
  starting_magnetization(1) = 0.3
  
  ! 约束磁化方向
  constrained_magnetization = 'atomic direction'
  angle1(1) = 0.0         ! 与z轴夹角 (度)
  angle2(1) = 0.0         ! xy平面投影与x轴夹角 (度)
  lambda = 0.5            ! 惩罚函数系数
  
  ! 或从共线计算启动
  lforcet = .true.        ! 旋转磁矩方向
/

&ELECTRONS
  mixing_beta = 0.1       ! SOC计算需要较小的mixing_beta
  startingpot = 'file'    ! 从文件读取势
/
```

**使用全相对论赝势**:

```
ATOMIC_SPECIES
Fe 55.845 Fe.rel-pbe-spn-rrkjus_psl.1.0.0.UPF
```

**注意**: 文件名中包含 `.rel.` 表示全相对论赝势

### 完整示例

```fortran
&CONTROL
   calculation = 'scf'
   prefix = 'fe_soc'
   outdir = './tmp/'
   pseudo_dir = '../pseudos/'
/

&SYSTEM
   ibrav = 3
   celldm(1) = 5.39
   nat = 1
   ntyp = 1
   
   ! SOC设置
   noncolin = .true.
   lspinorb = .true.
   
   ! 初始磁化
   starting_magnetization(1) = 0.3
   
   ! 计算参数
   ecutwfc = 70
   ecutrho = 850
   occupations = 'smearing'
   smearing = 'marzari-vanderbilt'
   degauss = 0.02
/

&ELECTRONS
   diagonalization = 'david'
   conv_thr = 1.0e-8
   mixing_beta = 0.7
/

ATOMIC_SPECIES
Fe 55.845 Fe.rel-pbe-spn-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
Fe 0.0 0.0 0.0

K_POINTS automatic
14 14 14 1 1 1
```

### DFT+U+SOC

```fortran
&SYSTEM
  noncolin = .true.
  lspinorb = .true.
  
  ! DFT+U设置
  lda_plus_u = .true.
  lda_plus_u_kind = 1     ! 必须设为1用于SOC
  Hubbard_U(1) = 4.0
/
```

**注意**: SOC计算必须使用 `lda_plus_u_kind = 1` (全局域极限)

### 能带计算 (含SOC)

```fortran
&CONTROL
   calculation = 'bands'
   prefix = 'fe_soc'
   outdir = './tmp/'
/

&SYSTEM
   ibrav = 3
   celldm(1) = 5.39
   nat = 1
   ntyp = 1
   noncolin = .true.
   lspinorb = .true.
   starting_magnetization(1) = 0.3
   ecutwfc = 70
   ecutrho = 850
   occupations = 'smearing'
   smearing = 'marzari-vanderbilt'
   degauss = 0.02
/

ATOMIC_SPECIES
Fe 55.845 Fe.rel-pbe-spn-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS alat
Fe 0.0 0.0 0.0

K_POINTS tpiba_b
6
0.000  0.000  0.000  40  ! Gamma
0.000  1.000  0.000  40  ! H
0.500  0.500  0.000  30  ! N
0.000  0.000  0.000  30  ! Gamma
0.500  0.500  0.500  30  ! P
0.000  1.000  0.000   1  ! H
```

**后处理** (bands.x输入):

```fortran
&BANDS
    outdir = './tmp/'
    prefix = 'fe_soc'
    filband = 'fe_bands_soc.dat'
    lsigma(3) = .true.      ! 计算z方向自旋期望值
/
```

---

## 关键参数总结

### VASP参数速查

| 参数 | 共线 | 非共线 | SOC |
|------|------|--------|-----|
| `ISPIN` | 2 | - | - |
| `LNONCOLLINEAR` | - | .TRUE. | 自动设置 |
| `LSORBIT` | - | - | .TRUE. |
| `MAGMOM` | 每离子1值 | 每离子3值 | 每离子3值 |
| `SAXIS` | - | 可选 | 重要 |
| `NBANDS` | N | 2N | 2N |
| `ISYM` | 默认 | 建议-1 | 建议-1 |
| `GGA_COMPAT` | - | - | .FALSE. |
| `LMAXMIX` | - | - | 4 (d), 6 (f) |

### QE参数速查

| 参数 | 说明 |
|------|------|
| `noncolin` | 非共线计算 |
| `lspinorb` | 自旋轨道耦合 |
| `starting_magnetization` | 初始磁化 |
| `angle1/angle2` | 磁矩角度约束 |
| `lambda` | 约束惩罚系数 |
| `constrained_magnetization` | 磁化约束类型 |
| `lforcet` | 从共线计算旋转启动 |

---

## 注意事项与最佳实践

### 收敛性

1. **EDIFF**: SOC/MAE计算需要极高精度 (1E-8 ~ 1E-9)
2. **k点**: 需要非常密集的k点网格
3. **展宽**: 使用小的SIGMA/degauss (0.01-0.05 eV)
4. **混合参数**: SOC计算可能需要调整AMIX/BMIX

### 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| "S matrix not positive definite" | 数值不稳定/原子重叠 | 增加ecutrho, 检查结构 |
| 不收敛 | 混合参数不当 | 减小mixing_beta |
| 磁矩不正确 | 初始MAGMOM不当 | 调整初始磁矩 |
| MAE符号错误 | 参考方向定义 | 检查SAXIS定义 |

### 计算建议

1. **分步计算**: 先共线 → 再非共线 → 最后SOC
2. **从CHGCAR启动**: 可大幅加速收敛
3. **对称性**: SOC计算建议关闭对称性 (ISYM=-1)
4. **赝势**: 使用全相对论赝势 (含 `.rel.`)

### 精度检查

```python
def check_mae_convergence():
    """检查MAE计算的收敛性"""
    # 1. 检查能量收敛
    assert EDIFF <= 1e-8, "EDIFF太大"
    
    # 2. 检查k点密度
    assert nk >= 16, "k点不够密集"
    
    # 3. 检查MAE数值合理性
    assert abs(MAE) < 10, "MAE值异常大"
    
    # 4. 检查磁矩
    assert abs(mag_moment - expected) < 0.5, "磁矩异常"
```

---

## 参考资源

- VASP Wiki: [LSORBIT](https://www.vasp.at/wiki/LSORBIT), [MAGMOM](https://www.vasp.at/wiki/MAGMOM), [Determining the Magnetic Anisotropy](https://vasp.at/wiki/Determining_the_Magnetic_Anisotropy)
- QE Tutorial: [Spin-Orbit Coupling](https://pranabdas.github.io/espresso/hands-on/soc/)
- 文献: Steiner et al., Phys. Rev. B 93, 224425 (2016)
