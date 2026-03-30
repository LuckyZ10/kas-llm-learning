# RPA响应函数方法 (Random Phase Approximation)

## 简介

随机相近似(RPA)是一种重要的多体微扰理论方法，用于计算电子相关能和响应函数。RPA通过考虑电子-空穴对的集体激发来描述系统的极化响应，在计算结合能、光学性质等方面具有重要应用。

## 理论基础

### ACFDT-RPA总能量

根据绝热连接涨落-耗散定理(ACFDT)，RPA总能量可表示为：

$$E_{\text{RPA}} = E_{\text{c}} + E_{\text{EXX}}$$

其中：
- $E_{\text{c}}$: RPA关联能
- $E_{\text{EXX}}$: Hartree-Fock交换能(使用DFT轨道计算)

### RPA关联能公式

$$E_{c} = \frac{1}{2\pi} \int_{0}^{\infty} \left[ \text{Tr} \ln(1-\chi(i\omega)V) + \chi(i\omega)V \right] d\omega$$

其中 $\chi(i\omega)$ 是独立粒子极化率，$V$ 是库仑相互作用。

---

## VASP实现

### 标准四步流程 (VASP 5.X)

#### Step 1: DFT计算

获取收敛的DFT波函数和电荷密度：

```
# INCAR
ENCUT = 600
EDIFF = 1E-8          # 高精度收敛
ISMEAR = 0
SIGMA = 0.05          # 小展宽，避免负占据
PREC = Accurate
LWAVE = .TRUE.
LCHARG = .TRUE.
```

**注意**: 
- 推荐使用PBE泛函
- 避免使用高阶Methfessel-Paxton展宽(ISMEAR > 0)
- EDIFF必须足够小以保证RPA能量收敛

#### Step 2: HF能量计算

计算Hartree-Fock能量：

```
# INCAR
ALGO = EIGENVAL
NELM = 1              # 仅计算能量，不自洽
LHFCALC = .TRUE.
AEXX = 1.0            # 纯HF，ALDAC自动设为0
LWAVE = .FALSE.       # 不更新WAVECAR
ISMEAR = 0
SIGMA = 0.05
```

**技巧**: 对于带隙较大的半导体/绝缘体，可设置 `HFRCUT = -1` 加速k点收敛。

#### Step 3: 计算空态

通过精确对角化获得足够数量的空轨道：

```
# INCAR - 从Step 1的OUTCAR中获取最大平面波数
NBANDS = [最大平面波数]    # vasp_gam需要乘以2
ALGO = Exact              # 精确对角化
EDIFF = 1E-8
LOPTICS = .TRUE.          # 金属体系建议设为.FALSE.
ISMEAR = 0
SIGMA = 0.05
LWAVE = .TRUE.
```

#### Step 4: RPA关联能计算

```
# INCAR
NBANDS = [最大平面波数]
ALGO = ACFDT            # 或 ALGO = RPA
NOMEGA = 12             # 虚频网格点数 (8-24)
ISMEAR = 0
SIGMA = 0.05
ENCUTGW = 400           # 响应函数截断能
ENCUTGWSOFT = 320       # 软截断 (VASP 6.3+ 默认为0.8*ENCUTGW)
```

**关键参数**:
- `NOMEGA`: 虚频/虚时网格点数
  - 大带隙绝缘体: 8
  - 半导体: 10-12
  - 金属: 12-24
- `ENCUTGW`: 响应函数截断能，决定计算精度

### 低标度RPA算法 (VASP 6.X)

对于大体系，使用基于虚时格林函数的算法：

```
# INCAR
ALGO = ACFDTR           # 或 ALGO = RPAR
NBANDS = [最大平面波数]
NOMEGA = 12
OMEGAMIN = [小于带隙的值]
OMEGATL = [大于最大跃迁能的值]
```

**算法原理**:
1. 在虚时轴上计算格林函数 $G(i\tau)$
2. 通过压缩公式计算极化率: $\chi(i\tau_m) = -G(i\tau_m)G(-i\tau_m)$
3. 压缩傅里叶变换到虚频空间
4. 计算关联能

### 有限温度RPA (VASP 6.1+)

适用于金属体系：

```
# INCAR
ALGO = ACFDT
LFINITE_TEMPERATURE = .TRUE.
ISMEAR = -1             # 必须使用Fermi展宽
SIGMA = 0.1             # 电子温度 (eV)
```

### 输出结果解读

OUTCAR中关键输出：

```
HF+RPA corr. energy TOTEN =    -xx.xxxxxxx eV
HF+E_corr(extrapolated)   =    -xx.xxxxxxx eV
HF energy                 =    -xx.xxxxxxx eV
RPA correlation energy    =     -x.xxxxxxx eV
```

- `TOTEN`: 当前截断能下的RPA总能量
- `extrapolated`: 外推到无限基组极限的能量
- VASP自动使用 $E_c(G) = E_c(\infty) + A/G^3$ 进行外推

---

## Quantum ESPRESSO实现

### epsilon.x 计算介电函数

QE通过 `epsilon.x` 计算RPA级别的复介电函数。

#### 输入文件准备

**Step 1: SCF计算** (`pw.scf.in`)

```fortran
&CONTROL
  calculation = 'scf'
  prefix = 'silicon'
  outdir = './tmp/'
  pseudo_dir = '../pseudos/'
  verbosity = 'high'
/

&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  nbnd = 20               ! 需要足够多的空带
  nosym = .TRUE.          ! 关闭对称性
  noinv = .TRUE.          ! 关闭反演对称性
/

&ELECTRONS
  mixing_beta = 0.6
  conv_thr = 1.0d-10
/

ATOMIC_SPECIES
  Si 28.086 Si.pz-vbc.UPF

ATOMIC_POSITIONS (alat)
  Si 0.0 0.0 0.0
  Si 0.25 0.25 0.25

K_POINTS automatic
6 6 6 0 0 0
```

**重要**: `epsilon.x` 不支持k点简并，必须设置 `nosym=.TRUE.` 和 `noinv=.TRUE.`

**Step 2: epsilon.x输入** (`epsilon.in`)

```fortran
&INPUTPP
  calculation = 'eps'     ! 计算复介电函数
  prefix = 'silicon'
  outdir = './tmp/'
/

&ENERGY_GRID
  smeartype = 'gaussian'  ! 或 'lorentzian'
  intersmear = 0.1        ! 带间展宽 (eV)
  intrasmear = 0.0        ! 带内展宽 (金属需设置 0.1-0.2 eV)
  wmin = 0.0              ! 最小频率 (eV)
  wmax = 30.0             ! 最大频率 (eV)
  nw = 500                ! 频率网格点数
  shift = 0.0             ! 刚性位移 (eV)
/
```

#### 计算类型选项

| calculation | 说明 |
|-------------|------|
| `eps` | 复介电函数 $\varepsilon(\omega) = \varepsilon_1 + i\varepsilon_2$ |
| `jdos` | 光学联合态密度 |

#### 运行命令

```bash
# 1. SCF计算
mpirun -np 16 pw.x -in pw.scf.in > pw.scf.out

# 2. 介电函数计算
mpirun -np 16 epsilon.x -in epsilon.in > epsilon.out
```

#### 输出文件

- `epsr_silicon.dat`: 介电函数实部 $\varepsilon_1$
- `epsi_silicon.dat`: 介电函数虚部 $\varepsilon_2$

文件格式：
```
# energy(eV) eps_xx eps_yy eps_zz
0.000  12.345  12.345  12.345
0.100  12.400  12.400  12.400
...
```

### turbo_eels.x (含局域场效应)

对于需要包含晶体局域场效应(CLFE)或激子效应的计算：

```fortran
&INPUTPP
  calculation = 'eels'
  prefix = 'system'
  outdir = './tmp/'
  approximation = 'RPA_with_CLFE'  ! 选项: TDDFT, IPA, RPA_with_CLFE
  q1 = 0.0001                      ! q矢量x分量
  q2 = 0.0
  q3 = 0.0
/
```

**approximation选项**:
- `IPA`: 独立粒子近似 (与epsilon.x类似)
- `TDDFT`: 含绝热LDA/GGA交换关联核
- `RPA_with_CLFE`: 含晶体局域场效应的RPA

---

## 关键参数总结

### VASP参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `ALGO` | 算法选择 | ACFDT/RPA (标准), ACFDTR/RPAR (低标度) |
| `NOMEGA` | 虚频网格数 | 8-24 (绝缘体8, 半导体10-12, 金属12-24) |
| `ENCUTGW` | 响应函数截断能 | 2/3 * ENCUT 或更高 |
| `ENCUTGWSOFT` | 软截断能 | 0.8 * ENCUTGW |
| `NBANDS` | 能带数 | 最大平面波数 |
| `OMEGAMIN/OMEGATL` | 跃迁能范围 | 自动/手动设置 |

### QE参数 (epsilon.x)

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `calculation` | 计算类型 | `eps` (介电函数), `jdos` (联合态密度) |
| `smeartype` | 展宽类型 | `gaussian` 或 `lorentzian` |
| `intersmear` | 带间展宽 | 0.05-0.2 eV |
| `intrasmear` | 带内展宽 | 0.0 (绝缘体), 0.1-0.2 (金属) |
| `wmin/wmax` | 频率范围 | 0-30 eV (根据需求调整) |
| `nw` | 频率点数 | 500-1000 |

---

## 后处理与分析

### 介电函数分析

**吸收系数计算**:

$$\alpha(\omega) = \frac{\omega}{c} \sqrt{\frac{-\varepsilon_1 + \sqrt{\varepsilon_1^2 + \varepsilon_2^2}}{2}}$$

**折射率计算**:

$$n(\omega) = \sqrt{\frac{\varepsilon_1 + \sqrt{\varepsilon_1^2 + \varepsilon_2^2}}{2}}$$

**反射率计算**:

$$R(\omega) = \left|\frac{1 - \sqrt{\varepsilon}}{1 + \sqrt{\varepsilon}}\right|^2$$

### Python后处理脚本

```python
import numpy as np
import matplotlib.pyplot as plt

# 读取epsilon.x输出
data = np.loadtxt('epsr_silicon.dat', skiprows=1)
energy = data[:, 0]
eps1 = data[:, 1]  # 取xx分量
eps2 = np.loadtxt('epsi_silicon.dat', skiprows=1)[:, 1]

# 计算光学性质
c = 299792458  # m/s
eV_to_J = 1.602e-19

# 吸收系数 (cm^-1)
alpha = (energy * eV_to_J / (6.626e-34 * c)) * \
        np.sqrt((-eps1 + np.sqrt(eps1**2 + eps2**2)) / 2) / 100

# 折射率
n = np.sqrt((eps1 + np.sqrt(eps1**2 + eps2**2)) / 2)

# 反射率
R = ((n - 1)**2 + (eps2/(2*n))**2) / ((n + 1)**2 + (eps2/(2*n))**2)

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(energy, eps1, label='$\\varepsilon_1$')
axes[0, 0].plot(energy, eps2, label='$\\varepsilon_2$')
axes[0, 0].set_xlabel('Energy (eV)')
axes[0, 0].set_ylabel('Dielectric Function')
axes[0, 0].legend()

axes[0, 1].plot(energy, alpha)
axes[0, 1].set_xlabel('Energy (eV)')
axes[0, 1].set_ylabel('Absorption Coefficient (cm$^{-1}$)')

axes[1, 0].plot(energy, n)
axes[1, 0].set_xlabel('Energy (eV)')
axes[1, 0].set_ylabel('Refractive Index')

axes[1, 1].plot(energy, R)
axes[1, 1].set_xlabel('Energy (eV)')
axes[1, 1].set_ylabel('Reflectivity')

plt.tight_layout()
plt.savefig('optical_properties.png', dpi=300)
```

---

## 注意事项与限制

### VASP限制

1. **基组收敛**: RPA关联能对截断能收敛很慢，需要外推
2. **k点收敛**: 需要密集的k点网格
3. **空态数**: 需要足够多的空带以获得收敛结果
4. **金属体系**: 需使用有限温度形式 (`LFINITE_TEMPERATURE`)

### QE限制 (epsilon.x)

1. **非局域势**: 不包含赝势非局域部分的贡献
2. **局域场效应**: 不包含晶体局域场效应
3. **激子效应**: 纯RPA，不包含激子相互作用
4. **USPP**: 超软赝势需要额外处理

### 改进方案

对于需要更高精度的计算：
- **VASP**: 使用BSE计算激子效应
- **QE**: 使用Yambo或TurboEELS包含局域场效应和激子
- **通用**: 使用BerkeleyGW进行完整的GW+BSE计算

---

## 参考资源

- VASP Wiki: [ACFDT/RPA calculations](https://www.vasp.at/wiki/index.php/ACFDT/RPA_calculations)
- QE Documentation: [epsilon.x](https://www.quantum-espresso.org/Doc/eps_man.tex)
- Yambo: [RPA and TDDFT calculations](http://www.yambo-code.org/)

## 文献推荐

1. Harl, Kresse, Phys. Rev. B 77, 045136 (2008) - RPA能量计算
2. Ren et al., Phys. Rev. B 86, 155140 (2012) - 低标度RPA
3. Kaltak, Kresse, Phys. Rev. B 101, 205145 (2020) - 有限温度GW/RPA
