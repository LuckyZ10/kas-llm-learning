# 电声耦合计算方法 (Electron-Phonon Coupling)

## 概述

电声耦合(Electron-Phonon Coupling, EPC)是描述电子与晶格振动(声子)相互作用的量子力学效应，对理解多种物理现象至关重要：

- **超导性**: 电声耦合是BCS超导理论的核心机制
- **电阻率**: 金属中电子被声子散射导致电阻
- **载流子迁移率**: 半导体中限制载流子输运的关键因素
- **热导率**: 电子-声子相互作用影响热输运
- **光谱线宽**: 拉曼光谱和ARPES中的峰展宽
- **温度依赖带隙**: 电子-声子相互作用导致带隙随温度变化

---

## 理论基础

### 电声耦合矩阵元

电声耦合矩阵元描述电子态 $|nk\rangle$ 与 $|mk+q\rangle$ 之间通过声子模式 $\nu$ 的跃迁：

$$g_{mn\nu}(k,q) = \sqrt{\frac{\hbar}{2M_\nu\omega_{q\nu}}} \langle mk+q | \partial_{q\nu}V^{scf} | nk \rangle$$

其中：
- $\partial_{q\nu}V^{scf}$: 自洽势对原子位移的导数
- $M_\nu$: 声子模式的有效质量
- $\omega_{q\nu}$: 声子频率

### Eliashberg谱函数

Eliashberg谱函数 $\alpha^2F(\omega)$ 描述特定频率声子与电子散射的概率：

$$\alpha^2F(\omega) = \frac{1}{N(\varepsilon_F)}\sum_{k,q,\nu} |g_{mn\nu}(k,q)|^2 \delta(\varepsilon_{nk}-\varepsilon_F)\delta(\varepsilon_{mk+q}-\varepsilon_F)\delta(\omega-\omega_{q\nu})$$

### 电声耦合常数

$$\lambda = 2\int_0^{\omega_{max}} \frac{\alpha^2F(\omega)}{\omega} d\omega$$

$\lambda$ 的物理意义：
- $\lambda < 0.3$: 弱耦合
- $0.3 < \lambda < 1$: 中等耦合
- $\lambda > 1$: 强耦合

### 超导临界温度 (McMillan公式)

$$T_c = \frac{\omega_{log}}{1.2} \exp\left[-\frac{1.04(1+\lambda)}{\lambda-\mu^*(1+0.62\lambda)}\right]$$

其中 $\omega_{log}$ 是对数平均频率，$\mu^*$ 是Coulomb赝势。

---

## Quantum ESPRESSO + EPW 计算

EPW (Electron-Phonon Wannier) 使用最大局域化Wannier函数插值，可在任意密集的k/q网格上高效计算电声耦合。

### 计算流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   pw.x scf  │───▶│  pw.x nscf  │───▶│   ph.x      │
│  (电子密度)  │    │ (波函数展开) │    │  (声子计算)  │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   epw.x     │◀───│  Wannier化  │◀───│  dvscf计算  │
│ (电声耦合)   │    │  (wannier90)│    │             │
└──────┬──────┘    └─────────────┘    └─────────────┘
       │
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Eliashberg  │    │  输运性质    │    │  超导性质    │
│ 谱函数计算   │    │  (电导/迁移率)│    │  (Tc计算)   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### 1. SCF计算

```fortran
&control
    calculation = 'scf'
    prefix = 'lead'
    outdir = './tmp'
    pseudo_dir = './pseudo'
/
&system
    ibrav = 2
    celldm(1) = 9.36
    nat = 1
    ntyp = 1
    ecutwfc = 60
    ecutrho = 480
    occupations = 'smearing'
    smearing = 'marzari-vanderbilt'
    degauss = 0.02
/
&electrons
    conv_thr = 1.0d-12
/
ATOMIC_SPECIES
    Pb 207.2 Pb.pbe-dn-kjpaw_psl.0.2.3.UPF
ATOMIC_POSITIONS alat
    Pb 0.0 0.0 0.0
K_POINTS automatic
    12 12 12 0 0 0
```

### 2. NSCF计算

```fortran
&control
    calculation = 'nscf'
    prefix = 'lead'
    outdir = './tmp'
/
&system
    ibrav = 2
    celldm(1) = 9.36
    nat = 1
    ntyp = 1
    ecutwfc = 60
    ecutrho = 480
    nbnd = 20
    occupations = 'smearing'
    smearing = 'marzari-vanderbilt'
    degauss = 0.02
/
&electrons
    conv_thr = 1.0d-12
/
ATOMIC_SPECIES
    Pb 207.2 Pb.pbe-dn-kjpaw_psl.0.2.3.UPF
ATOMIC_POSITIONS alat
    Pb 0.0 0.0 0.0
K_POINTS automatic
    6 6 6 0 0 0  ! 粗网格，EPW将插值到细网格
```

### 3. 声子计算 (ph.x)

```fortran
&inputph
    prefix = 'lead'
    outdir = './tmp'
    fildyn = 'lead.dyn'
    fildvscf = 'dvscf'
    ldisp = .true.
    nq1 = 4
    nq2 = 4
    nq3 = 4
    tr2_ph = 1.0d-16
/
```

### 4. EPW计算

```fortran
&inputepw
    prefix = 'lead'
    outdir = './tmp'
    dvscf_dir = './save'
    
    ! Wannier化设置
    wannierize = .true.
    num_iter = 300
    iprint = 2
    
    ! 投影轨道
    proj(1) = 'Pb:sp3'
    
    ! 粗网格 (DFT计算)
    nk1 = 6
    nk2 = 6
    nk3 = 6
    nq1 = 4
    nq2 = 4
    nq3 = 4
    
    ! 细网格 (Wannier插值)
    nkf1 = 40
    nkf2 = 40
    nkf3 = 40
    nqf1 = 20
    nqf2 = 20
    nqf3 = 20
    
    ! 费米面厚度
    fsthick = 1.0  ! eV
    
    ! 电声耦合计算
    elph = .true.
    ep_coupling = .true.
    
    ! Eliashberg函数
    a2f = .true.
    degaussw = 0.05  ! meV
    
    ! 超导计算
    eliashberg = .true.
    laniso = .true.
    limag = .true.
    lpade = .true.
    nsiter = 500
    conv_thr_iaxis = 1.0d-4
    wscut = 1.0  ! eV
    
    ! 温度网格
    nstemp = 50
    tempsmin = 1.0
    tempsmax = 20.0
    
    ! Coulomb赝势
    muc = 0.16
/
```

### EPW关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `nk1-3`/`nq1-3` | 粗k/q网格 (DFT) | 6×6×6 / 4×4×4 |
| `nkf1-3`/`nqf1-3` | 细k/q网格 (插值) | 40×40×40 / 20×20×20 |
| `fsthick` | 费米面厚度 | 0.3-1.0 eV |
| `degaussw` | 展宽参数 | 0.01-0.1 meV |
| `wscut` | 声子截断能量 | 0.5-1.0 eV |
| `muc` | Coulomb赝势 | 0.1-0.2 |

### 运行脚本

```bash
#!/bin/bash
# run_epw.sh - EPW计算完整流程

# 1. SCF计算
cd scf
mpirun -np 16 pw.x < scf.in > scf.out
cd ..

# 2. NSCF计算
cd nscf
cp -r ../scf/*.save ./
mpirun -np 16 pw.x < nscf.in > nscf.out
cd ..

# 3. 声子计算
cd phonon
cp -r ../scf/*.save ./
mpirun -np 16 ph.x < ph.in > ph.out
mpirun -np 16 q2r.x < q2r.in > q2r.out
mpirun -np 16 matdyn.x < matdyn.in > matdyn.out
cd ..

# 4. 准备EPW输入
cd epw
cp -r ../nscf/*.save ./
cp -r ../phonon/save ./
cp ../phonon/*.dyn ./

# 5. 运行EPW
mpirun -np 32 epw.x < epw.in > epw.out
```

---

## Yambo电声耦合计算

Yambo提供温度依赖的准粒子修正和Eliashberg函数计算。

### 计算流程

```
pw.x scf ──▶ pw.x nscf ──▶ p2y ──▶ yambo_ph setup ──▶ ph.x ──▶ dvscf计算
                                                              │
                                                              ▼
ypp_ph -g g ◀── yambo_ph el-ph ◀── ypp_ph import ◀── elph_dir/
```

### 1. QE准备计算

```fortran
! scf.in - 注意设置 force_symmorphic=.true.
&system
    ibrav = 0
    celldm(1) = 5.132
    force_symmorphic = .true.
    ...
/
CELL_PARAMETERS alat
0.0  1.0  1.0
1.0  0.0  1.0
1.0  1.0  0.0
```

### 2. 生成q点列表

```bash
# 进入nscf/save目录
cd nscf/si.save
p2y
yambo_ph -i  # 生成setup输入
yambo_ph     # 运行setup

# 生成q点列表
ypp_ph -k q  # 生成输入文件，设置ListPts和cooOut="alat"
ypp_ph       # 输出PW格式的q点列表
```

### 3. 声子计算

```fortran
&inputph
    prefix = 'si'
    fildvscf = 'si-dvscf'
    fildyn = 'si.dyn'
    electron_phonon = 'dvscf'
    epsil = .false.
    trans = .true.
    ldisp = .false.
    qplot = .true.
/
8  ! q点数量
    0.000000000  0.000000000  0.000000000 1
   -0.125000000 -0.125000000  0.125000000 1
    ...
```

### 4. 生成电声矩阵元

```fortran
&inputph
    prefix = 'si'
    fildvscf = 'si-dvscf'
    fildyn = 'si.dyn'
    electron_phonon = 'yambo'  ! 关键设置
    epsil = .false.
    trans = .false.
    ldisp = .false.
    qplot = .true.
/
```

### 5. 导入Yambo

```bash
cd dvscf/si.save
ypp_ph -g g  # 生成gkkp输入

# 编辑gkkp.in，设置DBsPATH指向elph_dir
ypp_ph       # 导入电声耦合数据
```

### 6. 温度依赖准粒子计算

```bash
# 生成输入
yambo_ph -g n -p fan -c ep -V gen
```

```fortran
dyson                            # [R] Dyson Equation solver
gw0                              # [R] GW approximation
el_ph_corr                       # [R] Electron-Phonon Correlation
Nelectro= 8.000000               # 电子数
ElecTemp= 0.000000         eV    # 电子温度
BoseTemp= 300.0            Kn    # 声子温度 (K)
OccTresh= 0.100000E-4            # 占据数阈值
SE_Threads=0                     # OpenMP线程数
DysSolver= "n"                   # Dyson方程求解器
% GphBRnge
  1 | 20 |                       # 电声耦合能带范围
%
% ElPhModes
  1 |  6 |                       # 包含的声子模式
%
GDamping= 0.0100000         eV   # Green函数展宽
RandQpts=0                       # 随机q点数
WRgFsq= .true.                   # 保存gFsq系数
%QPkrange                        # 准粒子k点/能带范围
1|8|1|12|
%
```

### 7. 后处理: Eliashberg函数

```bash
ypp_ph -s e  # 生成Eliashberg函数输入
```

```fortran
electrons                        # [R] Electronic properties
eliashberg                       # [R] Eliashberg
PhBroad= 0.0010000          eV    # 声子展宽
PhStps= 200                      # 能量步数
%QPkrange                        # k点/能带范围
 1|1|4|5|                        # 价带顶和导带底
%
```

---

## VASP电声耦合计算

VASP 6.5.0+ 支持基于线性化Boltzmann方程的输运系数计算，包含电子-声子散射。

### 输运系数计算

```fortran
PREC = Accurate
EDIFF = 1e-8
ISMEAR = -15
SIGMA = 0.01
LREAL = .FALSE.
LWAVE = .FALSE.
LCHARG = .FALSE.

# 电声耦合模式
ELPH_MODE = TRANSPORT

# 化学势确定
ELPH_ISMEAR = -15

# 输运系数计算
TRANSPORT_NEDOS = 501
ELPH_SELFEN_TEMPS = 0 100 200 300 400 500  ! 温度列表(K)
```

### 关键参数

| 参数 | 说明 |
|------|------|
| `ELPH_MODE` | 计算模式: TRANSPORT/SCATTERING |
| `ELPH_ISMEAR` | 电声计算展宽方法 |
| `TRANSPORT_NEDOS` | 能量网格点数 |
| `ELPH_SELFEN_TEMPS` | 计算温度的列表 |

---

## 结果分析

### Eliashberg函数绘图

```python
#!/usr/bin/env python3
"""
绘制Eliashberg谱函数 α²F(ω)
EPW输出文件: lead.a2f.01
"""
import numpy as np
import matplotlib.pyplot as plt

# 读取EPW输出的a2f文件
# 格式: 频率, 不同展宽的α²F(ω)...
data = np.loadtxt('lead.a2f.01')

omega = data[:, 0]  # 频率 (meV)
a2f_01 = data[:, 1]  # 展宽0.05 meV
a2f_02 = data[:, 2]  # 展宽0.10 meV

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图: Eliashberg函数
ax1 = axes[0]
ax1.plot(omega, a2f_01, 'b-', label='σ=0.05 meV')
ax1.plot(omega, a2f_02, 'r-', label='σ=0.10 meV')
ax1.set_xlabel('Frequency (meV)')
ax1.set_ylabel('α²F(ω)')
ax1.set_title('Eliashberg Spectral Function')
ax1.legend()
ax1.grid(True)

# 右图: 积分λ(ω)
ax2 = axes[1]
lambda_omega = 2 * np.cumsum(a2f_01 / omega) * np.diff(omega, prepend=0)
ax2.plot(omega, lambda_omega, 'g-')
ax2.axhline(y=lambda_omega[-1], color='r', linestyle='--', 
            label=f'λ = {lambda_omega[-1]:.3f}')
ax2.set_xlabel('Frequency (meV)')
ax2.set_ylabel('λ(ω)')
ax2.set_title('Electron-Phonon Coupling Constant')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('eliashberg.png', dpi=150)
```

### 温度依赖带隙分析

```python
#!/usr/bin/env python3
"""
分析Yambo输出的温度依赖准粒子修正
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 读取不同温度的QP输出
temperatures = [0, 50, 100, 150, 200, 250, 300]
gaps = []

for T in temperatures:
    # 读取o-T{temp}.qp文件
    data = np.loadtxt(f'o-T{T}.qp')
    # 计算带隙 (导带底 - 价带顶)
    vbm = data[data[:, 1] == 4, 2] + data[data[:, 1] == 4, 3]  # 实部+修正
    cbm = data[data[:, 1] == 5, 2] + data[data[:, 1] == 5, 3]
    gap = cbm[0] - vbm[0]
    gaps.append(gap)

# Varshni拟合
def varshni(T, Eg0, alpha, beta):
    """Varshni方程: Eg(T) = Eg0 - alpha*T^2/(T+beta)"""
    return Eg0 - alpha * T**2 / (T + beta)

popt, _ = curve_fit(varshni, temperatures, gaps, p0=[gaps[0], 0.0005, 300])

# 绘图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(temperatures, gaps, 'bo', label='Calculated')
T_fit = np.linspace(0, 350, 100)
ax.plot(T_fit, varshni(T_fit, *popt), 'r--', 
        label=f'Varshni fit: Eg(0)={popt[0]:.3f} eV')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Band Gap (eV)')
ax.set_title('Temperature-Dependent Band Gap')
ax.legend()
ax.grid(True)
plt.savefig('gap_vs_T.png', dpi=150)
```

### 声子线宽分析

```python
#!/usr/bin/env python3
"""
分析电声耦合导致的声子线宽
"""
import numpy as np
import matplotlib.pyplot as plt

# 从EPW输出读取声子线宽
# 文件格式: q点, 模式, 频率(meV), 线宽(meV)
data = np.loadtxt('linewidth.phself')

q_points = np.unique(data[:, 0])
n_modes = 6  # 对于单原子原胞

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 声子色散+线宽(颜色映射)
ax1 = axes[0]
for i in range(n_modes):
    mode_data = data[data[:, 1] == i+1]
    q = mode_data[:, 0]
    freq = mode_data[:, 2]
    linewidth = mode_data[:, 3]
    
    scatter = ax1.scatter(q, freq, c=linewidth, cmap='hot', 
                         s=20, vmin=0, vmax=0.5)

ax1.set_xlabel('q-point index')
ax1.set_ylabel('Frequency (meV)')
ax1.set_title('Phonon Dispersion (color = linewidth)')
plt.colorbar(scatter, ax=ax1, label='Linewidth (meV)')

# 线宽分布
ax2 = axes[1]
all_linewidths = data[:, 3]
ax2.hist(all_linewidths, bins=50, edgecolor='black')
ax2.set_xlabel('Linewidth (meV)')
ax2.set_ylabel('Count')
ax2.set_title('Phonon Linewidth Distribution')

plt.tight_layout()
plt.savefig('phonon_linewidth.png', dpi=150)
```

---

## 收敛性测试

### k/q网格收敛

```python
#!/usr/bin/env python3
"""
测试k/q网格对λ和Tc的收敛性
"""
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# 测试不同的细网格密度
nkf_list = [20, 30, 40, 50, 60]
nqf_list = [10, 15, 20, 25, 30]

lambda_values = []
tc_values = []

for nkf, nqf in zip(nkf_list, nqf_list):
    # 修改EPW输入文件
    with open('epw.in', 'r') as f:
        content = f.read()
    
    content = content.replace('nkf1 = .*', f'nkf1 = {nkf}')
    content = content.replace('nqf1 = .*', f'nqf1 = {nqf}')
    
    with open('epw.in', 'w') as f:
        f.write(content)
    
    # 运行EPW
    subprocess.run(['mpirun', '-np', '32', 'epw.x'], 
                   input=open('epw.in').read(), text=True)
    
    # 提取结果
    # 这里需要实现从输出文件中提取λ和Tc的代码
    lambda_val = extract_lambda('epw.out')
    tc_val = extract_tc('epw.out')
    
    lambda_values.append(lambda_val)
    tc_values.append(tc_val)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.plot(nkf_list, lambda_values, 'bo-')
ax1.set_xlabel('Fine k-grid (nkf)')
ax1.set_ylabel('λ')
ax1.set_title('Convergence of λ')
ax1.grid(True)

ax2 = axes[1]
ax2.plot(nkf_list, tc_values, 'rs-')
ax2.set_xlabel('Fine k-grid (nkf)')
ax2.set_ylabel('Tc (K)')
ax2.set_title('Convergence of Tc')
ax2.grid(True)

plt.tight_layout()
plt.savefig('convergence.png', dpi=150)
```

### 双网格方法 (Yambo)

双网格方法通过在粗网格周围使用细网格平均来加速收敛：

```bash
# 1. 生成细网格声子频率
matdyn.x < matdyn_fine.in > matdyn_fine.out

# 2. 读取双网格
ypp_ph -g d  # 生成gkkp_dg输入
# 设置 PHfreqF="si.freq_fine" 和 FineGd_mode="mixed"
ypp_ph       # 生成ndb.PH_Double_Grid

# 3. 使用双网格运行yambo_ph
yambo_ph -J T0_dg
```

---

## 常见问题与解决

### 1. 费米面采样不足

**症状**: λ值随网格增加剧烈变化

**解决**:
- 增加细网格密度 (nkf1-3)
- 调整费米面厚度 `fsthick`
- 使用随机网格 `rand_k=.true.`

### 2. 声子频率虚频

**症状**: 出现负的声子频率

**解决**:
- 检查结构优化是否充分
- 增加截断能
- 检查q点网格收敛性

### 3. Wannier化失败

**症状**: Wannier函数展宽过大或MLWF不收敛

**解决**:
- 调整投影轨道 `proj()`
- 增加冻结能窗 `dis_froz_max/min`
- 增加迭代次数 `num_iter`

### 4. 超导Tc计算不收敛

**症状**: Eliashberg方程迭代不收敛

**解决**:
- 增加迭代次数 `nsiter`
- 放宽收敛阈值 `conv_thr_iaxis`
- 调整Matsubara频率截断 `wscut`

### 5. 极性材料收敛慢

**症状**: 极性材料计算结果收敛极慢

**解决**:
- 使用LO-TO分裂修正
- 增加q点密度
- 考虑使用非解析项修正

---

## 参考文献

1. Ponce, S., et al. (2016). EPW: Electron-phonon coupling, transport and superconducting properties using maximally localized Wannier functions. *Computer Physics Communications*, 209, 116-133.

2. Giustino, F. (2017). Electron-phonon interactions from first principles. *Reviews of Modern Physics*, 89(1), 015003.

3. Marzari, N., et al. (2012). Maximally localized Wannier functions: Theory and applications. *Reviews of Modern Physics*, 84(4), 1419.

4. Allen, P. B., & Mitrovic, B. (1982). Theory of superconducting Tc. *Solid State Physics*, 37, 1-92.

5. McMillan, W. L. (1968). Transition temperature of strong-coupled superconductors. *Physical Review*, 167(2), 331.

6. Yambo Documentation: https://wiki.yambo-code.eu/

7. EPW Documentation: https://epw-code.org/

---

## 相关文档

- [热电输运](thermoelectric_transport.md) - BoltzWann和EPW输运计算
- [声子计算](calculation_methods.md#声子计算) - DFPT声子计算方法
- [GW近似](gw_approximation.md) - 电子自能计算
- [PIMD](pimd.md) - 路径积分分子动力学
