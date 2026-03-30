# 热电输运计算方法 (Thermoelectric Transport)

## 概述

热电输运研究电子和热量在材料中的传输行为，对以下应用至关重要：

- **热电发电**: 将废热转化为电能 (Seebeck效应)
- **热电制冷**: 通过电流实现制冷 (Peltier效应)
- **热管理**: 电子器件散热设计
- **能源材料**: 高效热电材料设计 (ZT值优化)

### 关键物理量

| 物理量 | 符号 | 单位 | 物理意义 |
|--------|------|------|----------|
| 电导率 | σ | S/m 或 1/Ω/m | 电子传导能力 |
| Seebeck系数 | S | V/K | 温差产生电压的效率 |
| 热导率 | κ | W/m/K | 热量传导能力 |
| 功率因子 | PF | μW/cm/K² | S²σ，热电性能指标 |
| 品质因数 | ZT | 无量纲 | S²σT/κ，综合性能指标 |

---

## 理论基础

### Boltzmann输运方程

在弛豫时间近似下，电流密度和热流密度可表示为：

$$\mathbf{J} = \mathbf{\sigma}(\mathbf{E} - \mathbf{S}\nabla T)$$

$$\mathbf{J}_Q = T\mathbf{\sigma}\mathbf{S}\mathbf{E} - \mathbf{K}\nabla T$$

其中输运系数通过对输运分布函数(TDF)积分得到：

$$\sigma_{ij}(\mu,T) = e^2 \int_{-\infty}^{+\infty} d\varepsilon \left(-\frac{\partial f}{\partial \varepsilon}\right) \Sigma_{ij}(\varepsilon)$$

$$[\mathbf{\sigma}\mathbf{S}]_{ij}(\mu,T) = \frac{e}{T} \int_{-\infty}^{+\infty} d\varepsilon \left(-\frac{\partial f}{\partial \varepsilon}\right)(\varepsilon-\mu) \Sigma_{ij}(\varepsilon)$$

$$K_{ij}(\mu,T) = \frac{1}{T} \int_{-\infty}^{+\infty} d\varepsilon \left(-\frac{\partial f}{\partial \varepsilon}\right)(\varepsilon-\mu)^2 \Sigma_{ij}(\varepsilon)$$

### 输运分布函数 (TDF)

$$\Sigma_{ij}(\varepsilon) = \frac{1}{V} \sum_{n,\mathbf{k}} v_i(n,\mathbf{k}) v_j(n,\mathbf{k}) \tau(n,\mathbf{k}) \delta(\varepsilon - E_{n,\mathbf{k}})$$

其中：
- $v_i(n,\mathbf{k}) = \frac{1}{\hbar}\frac{\partial E_{n,\mathbf{k}}}{\partial k_i}$: 群速度
- $\tau$: 弛豫时间 (散射时间的平均值)
- $f$: Fermi-Dirac分布函数

### 热导率与电导率关系

电子热导率与电导率通过Wiedemann-Franz定律关联：

$$\kappa_e = L \sigma T$$

其中Lorenz数 $L = \frac{\pi^2}{3}(\frac{k_B}{e})^2 \approx 2.44 \times 10^{-8}$ WΩ/K² (金属极限)

总热导率：$\kappa = \kappa_e + \kappa_l$ (电子+晶格贡献)

---

## BoltzWann计算 (Wannier90)

BoltzWann使用最大局域化Wannier函数和Boltzmann输运方程计算热电输运性质。

### 计算流程

```
pw.x scf ──▶ pw.x nscf ──▶ wannier90.x ──▶ postw90.x (BoltzWann)
```

### 1. SCF计算

```fortran
&control
    calculation = 'scf'
    prefix = 'si'
    outdir = './tmp'
/
&system
    ibrav = 2
    celldm(1) = 10.26
    nat = 2
    ntyp = 1
    ecutwfc = 40
    ecutrho = 320
    occupations = 'smearing'
    smearing = 'cold'
    degauss = 0.02
/
&electrons
    conv_thr = 1.0d-10
/
ATOMIC_SPECIES
    Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS alat
    Si 0.00 0.00 0.00
    Si 0.25 0.25 0.25
K_POINTS automatic
    12 12 12 0 0 0
```

### 2. NSCF计算 (密集k点)

```fortran
&control
    calculation = 'nscf'
    prefix = 'si'
    outdir = './tmp'
/
&system
    ibrav = 2
    celldm(1) = 10.26
    nat = 2
    ntyp = 1
    ecutwfc = 40
    nbnd = 20
    occupations = 'smearing'
    smearing = 'cold'
    degauss = 0.02
/
&electrons
    conv_thr = 1.0d-10
/
ATOMIC_SPECIES
    Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS alat
    Si 0.00 0.00 0.00
    Si 0.25 0.25 0.25
K_POINTS automatic
    20 20 20 0 0 0  ! 密集k点用于Wannier化
```

### 3. Wannier90输入

```fortran
! si.win - Wannier90输入文件
num_wann = 8
num_iter = 100

! 能窗
dis_win_min = -5.0
dis_win_max = 10.0
dis_froz_min = -3.0
dis_froz_max = 3.0

! 投影轨道
begin projections
Si: sp3
end projections

! k点网格
mp_grid = 20 20 20

begin kpoints
...  ! 从pw.x输出获取k点列表
end kpoints

! BoltzWann设置
boltzwann = true
boltz_kmesh = 100 100 100  ! 输运计算用的密集k网格

! 温度范围
boltz_temp_min = 100.0
boltz_temp_max = 1000.0
boltz_temp_step = 100.0

! 化学势范围
boltz_mu_min = -2.0
boltz_mu_max = 2.0
boltz_mu_step = 0.1

! 弛豫时间 (fs)
boltz_relax_time = 10.0

! 计算DOS
boltz_calc_also_dos = true
```

### 4. 运行流程

```bash
#!/bin/bash
# run_boltzwann.sh

# 1. SCF计算
cd scf
mpirun -np 16 pw.x < scf.in > scf.out
cd ..

# 2. NSCF计算
cd nscf
cp -r ../scf/*.save ./
mpirun -np 16 pw.x < nscf.in > nscf.out
# 生成Wannier90的k点列表
../utils/kpoints2wannier.py ../scf/si.save/K_POINTS
wannier90.x -pp si  # 预处理，生成si.nnkp
cd ..

# 3. Wannier化
cd wannier
cp ../nscf/si.save/charge-density.dat ./
cp ../nscf/si.save/wfc*.dat ./
pw2wannier90.x < pw2wan.in > pw2wan.out
wannier90.x si  # 生成si.eig, si.mmn, si.amn
cd ..

# 4. BoltzWann计算
cd boltzwann
cp ../wannier/si.eig ./
cp ../wannier/si.mmn ./
cp ../wannier/si.amn ./
postw90.x si  # 运行BoltzWann
cd ..
```

### BoltzWann关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `boltz_kmesh` | 输运计算k点网格 | 100×100×100 或更高 |
| `boltz_relax_time` | 弛豫时间 (fs) | 10-50 fs |
| `boltz_temp_min/max` | 温度范围 (K) | 100-1000 K |
| `boltz_mu_min/max` | 化学势范围 (eV) | 带隙范围内 |
| `boltz_dos_smr_type` | DOS展宽类型 | 'gauss' 或 'm-p' |
| `boltz_dos_smr_fixed_en_width` | 固定展宽宽度 (eV) | 0.01-0.05 eV |

### 输出文件解析

```bash
# 生成的文件:
# si_tdf.dat       - 输运分布函数
# si_elcond.dat    - 电导率张量
# si_seebeck.dat   - Seebeck系数张量
# si_sigmas.dat    - σS张量
# si_kappa.dat     - K张量
# si_boltzdos.dat  - DOS
```

---

## EPW输运计算

EPW提供基于第一性原理电声耦合的输运性质计算，包括电阻率、迁移率和热导率。

### 金属电阻率计算 (Ziman公式)

对于金属，电阻率可通过Eliashberg输运谱函数计算：

$$\rho(T) = \frac{4\pi m_e}{n e^2 k_B T} \int_0^{\infty} d\omega \hbar\omega \alpha_{tr}^2 F(\omega) n(\omega,T)[1+n(\omega,T)]$$

### EPW输运计算输入

```fortran
&inputepw
    prefix = 'lead'
    outdir = './tmp'
    dvscf_dir = './save'
    
    ! Wannier化
    wannierize = .true.
    num_iter = 300
    proj(1) = 'Pb:sp3'
    
    ! 粗网格
    nk1 = 6
    nk2 = 6
    nk3 = 6
    nq1 = 4
    nq2 = 4
    nq3 = 4
    
    ! 细网格
    nkf1 = 60
    nkf2 = 60
    nkf3 = 60
    nqf1 = 30
    nqf2 = 30
    nqf3 = 30
    
    ! 费米面设置
    fsthick = 1.0  ! eV
    degaussw = 0.05  ! meV
    
    ! 输运计算
    phonselfen = .true.
    a2f = .true.
    
    ! 电阻率计算 (Ziman公式)
    resistivity = .true.
    
    ! 温度范围
    nstemp = 100
    tempsmin = 10.0
    tempsmax = 1000.0
/
```

### EPW输运输出

```
# 输出文件:
# lead.res.01       - 电阻率随温度变化
# lead.a2f.tr.01    - 输运Eliashberg谱函数
# lead.sigmap.01    - 电导率
```

### 半导体迁移率计算

EPW 5.0+ 支持基于BTE的迁移率计算：

```fortran
&inputepw
    prefix = 'silicon'
    
    ! 半导体特定设置
    assume_metal = .false.
    
    ! 载流子浓度
    nc = 1.0d18  ! cm^-3，电子浓度
    nv = 0.0     ! 空穴浓度
    
    ! 输运计算
    mob_maxiter = 100
    mob_conv_thr = 1.0d-6
    
    ! 插值设置
    nkf1 = 80
    nkf2 = 80
    nkf3 = 80
/
```

---

## VASP输运计算

VASP 6.5.0+ 支持基于电子-声子散射的输运系数计算。

### 输入文件 (INCAR)

```fortran
PREC = Accurate
EDIFF = 1e-8
ISMEAR = -15
SIGMA = 0.01
LREAL = .FALSE.
LWAVE = .FALSE.
LCHARG = .FALSE.

# 电声耦合输运模式
ELPH_MODE = TRANSPORT

# 化学势确定
ELPH_ISMEAR = -15

# 输运系数计算
TRANSPORT_NEDOS = 501
ELPH_SELFEN_TEMPS = 0 100 200 300 400 500

# 截断能
ENCUT = 500

# k点密度 (高对称性方向)
KPOINTS = 50 50 50
```

### 计算步骤

```bash
#!/bin/bash
# run_vasp_transport.sh

# 1. 结构优化
mpirun -np 32 vasp_std

# 2. 静态计算 (获取波函数)
cp CONTCAR POSCAR
mpirun -np 32 vasp_std

# 3. 声子计算 (如果需要)
# 使用PHONOPY或VASP的IBRION=7/8

# 4. 输运计算
# 修改INCAR添加ELPH_MODE = TRANSPORT
mpirun -np 32 vasp_std
```

---

## 结果分析与可视化

### 电导率与温度关系

```python
#!/usr/bin/env python3
"""
分析BoltzWann输出的电导率数据
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 读取电导率数据
# 格式: mu(eV), T(K), sigma_xx, sigma_xy, sigma_yy, sigma_xz, sigma_yz, sigma_zz
data = np.loadtxt('si_elcond.dat', comments='#')

mu = data[:, 0]  # 化学势
temp = data[:, 1]  # 温度
sigma_xx = data[:, 2]  # xx分量

# 创建网格用于等高线图
mu_grid = np.linspace(mu.min(), mu.max(), 200)
temp_grid = np.linspace(temp.min(), temp.max(), 200)
MU, TEMP = np.meshgrid(mu_grid, temp_grid)

# 插值
SIGMA = griddata((mu, temp), sigma_xx, (MU, TEMP), method='cubic')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图: 等高线图
ax1 = axes[0]
levels = np.logspace(np.log10(sigma_xx[sigma_xx > 0].min()), 
                     np.log10(sigma_xx.max()), 20)
contour = ax1.contourf(MU, TEMP, SIGMA, levels=levels, cmap='viridis')
ax1.set_xlabel('Chemical Potential (eV)')
ax1.set_ylabel('Temperature (K)')
ax1.set_title('Electrical Conductivity σ_xx (S/m)')
plt.colorbar(contour, ax=ax1, label='σ (S/m)')

# 标记带边位置
ax1.axvline(x=0.5, color='r', linestyle='--', label='CBM')
ax1.axvline(x=-0.5, color='b', linestyle='--', label='VBM')
ax1.legend()

# 右图: 固定化学势下的温度依赖
ax2 = axes[1]
# 选择本征费米能级附近的电导率
intrinsic_sigma = sigma_xx[(np.abs(mu) < 0.1)]
intrinsic_temp = temp[(np.abs(mu) < 0.1)]

# 按温度排序
sort_idx = np.argsort(intrinsic_temp)
ax2.semilogy(intrinsic_temp[sort_idx], intrinsic_sigma[sort_idx], 'bo-', label='Intrinsic')

# n型掺杂 (化学势在导带中)
n_type_sigma = sigma_xx[(mu > 0.5) & (mu < 0.7)]
n_type_temp = temp[(mu > 0.5) & (mu < 0.7)]
sort_idx = np.argsort(n_type_temp)
ax2.semilogy(n_type_temp[sort_idx], n_type_sigma[sort_idx], 'rs-', label='n-type')

ax2.set_xlabel('Temperature (K)')
ax2.set_ylabel('Conductivity (S/m)')
ax2.set_title('Temperature-Dependent Conductivity')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('conductivity_analysis.png', dpi=150)
```

### Seebeck系数分析

```python
#!/usr/bin/env python3
"""
分析Seebeck系数并计算功率因子
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 读取数据
data_s = np.loadtxt('si_seebeck.dat', comments='#')
data_sigma = np.loadtxt('si_elcond.dat', comments='#')

mu = data_s[:, 0]
temp = data_s[:, 1]
S_xx = data_s[:, 2]  # Seebeck系数 (V/K)
sigma_xx = data_sigma[:, 2]  # 电导率 (S/m)

# 计算功率因子 PF = S²σ (单位: μW/cm/K²)
PF = (S_xx * 1e6)**2 * sigma_xx / 100  # 转换为 μW/cm/K²

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Seebeck系数
ax1 = axes[0]
# 选择300K的数据
temp_300 = (np.abs(temp - 300) < 1)
ax1.plot(mu[temp_300], S_xx[temp_300] * 1e6, 'b-', linewidth=2)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
ax1.set_xlabel('Chemical Potential (eV)')
ax1.set_ylabel('Seebeck Coefficient (μV/K)')
ax1.set_title('S @ 300K')
ax1.grid(True, alpha=0.3)

# 电导率
ax2 = axes[1]
ax2.semilogy(mu[temp_300], sigma_xx[temp_300], 'r-', linewidth=2)
ax2.set_xlabel('Chemical Potential (eV)')
ax2.set_ylabel('Conductivity (S/m)')
ax2.set_title('σ @ 300K')
ax2.grid(True, alpha=0.3)

# 功率因子
ax3 = axes[2]
ax3.plot(mu[temp_300], PF[temp_300], 'g-', linewidth=2)
ax3.set_xlabel('Chemical Potential (eV)')
ax3.set_ylabel('Power Factor (μW/cm/K²)')
ax3.set_title('PF @ 300K')
ax3.grid(True, alpha=0.3)

# 标记最优掺杂位置
max_pf_idx = np.argmax(PF[temp_300])
optimal_mu = mu[temp_300][max_pf_idx]
ax3.axvline(x=optimal_mu, color='r', linestyle='--', 
            label=f'Optimal μ = {optimal_mu:.2f} eV')
ax3.legend()

plt.tight_layout()
plt.savefig('thermoelectric_properties.png', dpi=150)

print(f"Maximum Power Factor @ 300K: {PF[temp_300].max():.2f} μW/cm/K²")
print(f"Optimal Chemical Potential: {optimal_mu:.3f} eV")
```

### 电阻率温度依赖 (EPW)

```python
#!/usr/bin/env python3
"""
分析EPW计算的电阻率随温度变化
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 读取EPW电阻率输出
# 格式: 温度(K), 展宽(meV), 电阻率(μΩ·cm)
data = np.loadtxt('lead.res.01')

temp = data[:, 0]
resistivity = data[:, 1]  # 选择特定展宽的数据

# Bloch-Grüneisen拟合
def bloch_gruneisen(T, rho_0, A, theta_D):
    """
    Bloch-Grüneisen公式:
    ρ(T) = ρ_0 + A*(T/θ_D)^5 * ∫[0 to θ_D/T] x^5/(e^x-1)/(1-e^-x) dx
    简化形式 (高温): ρ(T) ≈ ρ_0 + A*T
    """
    from scipy.integrate import quad
    
    def integrand(x):
        return x**5 / ((np.exp(x) - 1) * (1 - np.exp(-x)))
    
    rho = np.zeros_like(T)
    for i, t in enumerate(T):
        if t > 0:
            integral, _ = quad(integrand, 0, theta_D/t)
            rho[i] = rho_0 + A * (t/theta_D)**5 * integral
    return rho

# 高温线性拟合
def linear(T, rho_0, alpha):
    return rho_0 + alpha * T

# 拟合
popt, _ = curve_fit(linear, temp[temp > 100], resistivity[temp > 100])

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(temp, resistivity, 'bo', markersize=6, label='Calculated')
ax.plot(temp, linear(temp, *popt), 'r--', 
        label=f'Fit: ρ = {popt[0]:.2f} + {popt[1]:.4f}·T')

ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Resistivity (μΩ·cm)')
ax.set_title('Electrical Resistivity of Pb')
ax.legend()
ax.grid(True, alpha=0.3)

# 计算残余电阻率比 (RRR)
rho_300 = np.interp(300, temp, resistivity)
rho_0 = popt[0]  # 外推到0K
rrr = rho_300 / rho_0
ax.text(0.5, 0.9, f'RRR = {rrr:.1f}', transform=ax.transAxes, 
        fontsize=12, verticalalignment='top')

plt.savefig('resistivity_temperature.png', dpi=150)

print(f"Residual Resistivity (0K): {rho_0:.2f} μΩ·cm")
print(f"Resistivity @ 300K: {rho_300:.2f} μΩ·cm")
print(f"RRR: {rrr:.1f}")
```

### ZT值计算

```python
#!/usr/bin/env python3
"""
计算热电品质因数 ZT = S²σT/κ
"""
import numpy as np
import matplotlib.pyplot as plt

# 读取输运数据
data_s = np.loadtxt('si_seebeck.dat', comments='#')
data_sigma = np.loadtxt('si_elcond.dat', comments='#')
data_kappa = np.loadtxt('si_kappa.dat', comments='#')

mu = data_s[:, 0]
temp = data_s[:, 1]
S = data_s[:, 2]  # V/K
sigma = data_sigma[:, 2]  # S/m
K = data_kappa[:, 2]  # W/m/K (这是K张量，不是κ)

# 计算热导率 κ = K - S²σT
# 注意: K是代码输出的张量，κ是真正的热导率
kappa = K - S**2 * sigma * temp

# 添加晶格热导率贡献 (从其他计算获得)
kappa_lattice = 50.0  # W/m/K (示例值)
kappa_total = kappa + kappa_lattice

# 计算ZT
ZT = S**2 * sigma * temp / kappa_total

# 选择特定温度
target_temp = 800  # K
temp_mask = (np.abs(temp - target_temp) < 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ZT vs 化学势
ax1 = axes[0]
ax1.plot(mu[temp_mask], ZT[temp_mask], 'b-', linewidth=2)
ax1.set_xlabel('Chemical Potential (eV)')
ax1.set_ylabel('ZT')
ax1.set_title(f'Figure of Merit @ {target_temp}K')
ax1.grid(True, alpha=0.3)

# 标记最大ZT
max_zt_idx = np.argmax(ZT[temp_mask])
optimal_mu = mu[temp_mask][max_zt_idx]
max_zt = ZT[temp_mask][max_zt_idx]
ax1.axvline(x=optimal_mu, color='r', linestyle='--')
ax1.scatter([optimal_mu], [max_zt], color='r', s=100, zorder=5)
ax1.text(optimal_mu, max_zt, f'  ZT={max_zt:.2f}', fontsize=10)

# ZT vs 温度 (最优掺杂)
ax2 = axes[1]
# 找到每个温度下的最大ZT
unique_temps = np.unique(temp)
max_zt_vs_temp = []
for t in unique_temps:
    t_mask = (np.abs(temp - t) < 1)
    max_zt_vs_temp.append(np.max(ZT[t_mask]))

ax2.plot(unique_temps, max_zt_vs_temp, 'ro-', markersize=6)
ax2.set_xlabel('Temperature (K)')
ax2.set_ylabel('Maximum ZT')
ax2.set_title('ZT vs Temperature (Optimal Doping)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('zt_analysis.png', dpi=150)

print(f"Maximum ZT @ {target_temp}K: {max_zt:.2f}")
print(f"Optimal Chemical Potential: {optimal_mu:.3f} eV")
```

---

## 弛豫时间模型

### 常数弛豫时间近似

最简单的近似，假设$\tau$与能量和温度无关：

```fortran
! BoltzWann输入
boltz_relax_time = 10.0  ! fs
```

### 能量依赖弛豫时间

对于声学声子散射主导的半导体：

$$\tau(E) = \tau_0 (E/k_B T)^r$$

其中 $r = -1/2$ (声学声子散射) 或 $r = 3/2$ (电离杂质散射)

### 温度依赖弛豫时间

```python
#!/usr/bin/env python3
"""
实现能量和温度依赖的弛豫时间模型
"""
import numpy as np

def relaxation_time(E, T, tau_0=10e-15, E_ref=0.025, r=-0.5, T_exp=-1.5):
    """
    能量和温度依赖的弛豫时间模型
    
    参数:
    E: 能量 (eV)
    T: 温度 (K)
    tau_0: 参考弛豫时间 (s)
    E_ref: 参考能量 (eV, 默认k_B*300K)
    r: 能量指数
    T_exp: 温度指数
    """
    k_B = 8.617e-5  # eV/K
    tau = tau_0 * (E / E_ref)**r * (T / 300)**T_exp
    return tau

# 声学声子散射
def acoustic_phonon_scattering(E, T, tau_0=10e-15):
    """声学声子散射: τ ∝ E^(-1/2) T^(-1)"""
    return relaxation_time(E, T, tau_0, r=-0.5, T_exp=-1.0)

# 光学声子散射
def optical_phonon_scattering(E, T, tau_0=5e-15, E_opt=0.08):
    """光学声子散射: 需要光学声子能量"""
    k_B = 8.617e-5
    if E > E_opt:
        return tau_0 * (np.exp(E_opt/(k_B*T)) - 1)
    else:
        return np.inf

# 电离杂质散射
def ionized_impurity_scattering(E, T, tau_0=50e-15):
    """电离杂质散射: τ ∝ E^(3/2) T^(3/2)"""
    return relaxation_time(E, T, tau_0, r=1.5, T_exp=1.5)
```

---

## 收敛性测试

### k点网格收敛

```python
#!/usr/bin/env python3
"""
测试k点网格对输运性质的收敛性
"""
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# 测试不同k点密度
kmesh_list = [50, 75, 100, 125, 150]
sigma_values = []
S_values = []

for kmesh in kmesh_list:
    # 修改Wannier90输入
    with open('si.win', 'r') as f:
        lines = f.readlines()
    
    with open('si.win', 'w') as f:
        for line in lines:
            if 'boltz_kmesh' in line:
                f.write(f'boltz_kmesh = {kmesh} {kmesh} {kmesh}\n')
            else:
                f.write(line)
    
    # 运行postw90
    subprocess.run(['postw90.x', 'si'])
    
    # 提取结果 (在特定化学势和温度)
    data = np.loadtxt('si_elcond.dat', comments='#')
    sigma_xx = data[0, 2]  # 第一个点的xx分量
    sigma_values.append(sigma_xx)
    
    data_s = np.loadtxt('si_seebeck.dat', comments='#')
    S_xx = data_s[0, 2]
    S_values.append(S_xx)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.plot(kmesh_list, sigma_values, 'bo-')
ax1.set_xlabel('k-mesh density')
ax1.set_ylabel('Conductivity (S/m)')
ax1.set_title('Convergence of Conductivity')
ax1.grid(True)

ax2 = axes[1]
ax2.plot(kmesh_list, np.array(S_values)*1e6, 'rs-')
ax2.set_xlabel('k-mesh density')
ax2.set_ylabel('Seebeck Coefficient (μV/K)')
ax2.set_title('Convergence of Seebeck Coefficient')
ax2.grid(True)

plt.tight_layout()
plt.savefig('convergence_kmesh.png', dpi=150)
```

---

## 常见问题

### 1. Seebeck系数符号错误

**症状**: n型材料显示正Seebeck系数

**原因**: 
- 能带顺序错误
- 化学势参考点设置不当

**解决**:
- 检查Wannier插值的能带结构
- 确认价带顶在0 eV

### 2. 电导率数值异常

**症状**: 电导率比实验值大/小几个数量级

**原因**:
- 弛豫时间设置不当
- k点网格不够密集

**解决**:
- 根据实验或文献调整`boltz_relax_time`
- 增加`boltz_kmesh`密度

### 3. 带隙区域电导率不为零

**症状**: 带隙中电导率不为零

**原因**:
- DOS展宽过大
- 能带交叉或Wannier化质量差

**解决**:
- 减小`boltz_dos_smr_fixed_en_width`
- 检查Wannier函数局域性

### 4. 金属电阻率计算发散

**症状**: EPW电阻率计算出现NaN或极大值

**原因**:
- 费米面采样不足
- 温度过低导致数值不稳定

**解决**:
- 增加`nkf`网格密度
- 从较高温度开始计算

---

## 参考文献

1. Pizzi, G., et al. (2014). BoltzWann: A code for the evaluation of thermoelectric and electronic transport properties with a maximally-localized Wannier functions basis. *Computer Physics Communications*, 185, 422-429.

2. Ponce, S., et al. (2016). EPW: Electron-phonon coupling, transport and superconducting properties using maximally localized Wannier functions. *Computer Physics Communications*, 209, 116-133.

3. Madsen, G. K., & Singh, D. J. (2006). BoltzTraP: A code for calculating band-structure dependent quantities. *Computer Physics Communications*, 175(1), 67-71.

4. Ziman, J. M. (1960). *Electrons and Phonons: The Theory of Transport Phenomena in Solids*. Oxford University Press.

5. Ashcroft, N. W., & Mermin, N. D. (1976). *Solid State Physics*. Holt, Rinehart and Winston.

6. Snyder, G. J., & Toberer, E. S. (2008). Complex thermoelectric materials. *Nature Materials*, 7(2), 105-114.

---

## 相关文档

- [电声耦合](electron_phonon_coupling.md) - EPW/Yambo电声耦合计算
- [GW近似](gw_approximation.md) - 电子自能计算
- [能带计算](calculation_methods.md#能带计算) - 基础电子结构
