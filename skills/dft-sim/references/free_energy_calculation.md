# 自由能计算方法 (FEP/TI/TI-DPS)

## 1. 理论基础

### 1.1 自由能重要性
自由能是判断热力学稳定性和相变的关键量：
- **Gibbs自由能 (G)**: 恒温恒压过程
- **Helmholtz自由能 (F)**: 恒温恒容过程
- **化学势**: 粒子交换平衡

### 1.2 计算方法对比

| 方法 | 全称 | 适用场景 | 精度 | 计算成本 |
|------|------|----------|------|----------|
| **FEP** | Free Energy Perturbation | 小扰动 | 中 | 低 |
| **TI** | Thermodynamic Integration | 连续路径 | 高 | 中 |
| **BAR/MBAR** | Bennett Acceptance Ratio | 双向采样 | 很高 | 中 |
| **TP** | Temperature Integration | 温度扫描 | 高 | 高 |
| **Alchemical** | 炼金变换 | 物种变换 | 高 | 中 |

### 1.3 热力学积分核心公式

**Helmholtz自由能变化**:
```
ΔF = ∫₀¹ ⟨∂H/∂λ⟩_λ dλ
```

**Gibbs自由能 (恒温恒压)**:
```
ΔG = ΔF + ∫ P dV ≈ ΔF + PΔV  (对凝聚相)
```

**炼金自由能** (化学势差):
```
ΔG = -kT ln⟨exp(-ΔU/kT)⟩₀  (FEP, Zwanzig方程)
```

---

## 2. VASP 自由能计算

### 2.1 慢增长法 (Slow Growth TI)

```bash
mkdir ti_calc && cd ti_calc
```

**INCAR** (热力学积分):
```
SYSTEM = TI Slow Growth

# 基础设置
ENCUT = 400
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.1

# 分子动力学
MDALGO = 2          ! Nose-Hoover恒温器
SMASS = 0           ! 无阻尼Nose-Hoover
TEBEG = 300         ! 起始温度 (K)
TEEND = 300         ! 结束温度 (K)
POTIM = 1.0         ! 时间步长 (fs)
NSW = 5000          ! MD步数

# 热力学积分关键参数
LINTERFAST = .TRUE. ! 启用TI
LDNOSCALE = .TRUE.  ! 关闭速度重标
LSCLLINE = .TRUE.   ! 线性标度

# 反应坐标 (λ)
# 在POSCAR中定义初始和终态结构
# VASP会通过LINEAR_COUPLING自动插值
```

**POSCAR** (双端点格式):
```
TiO2 Phase Transition
1.0
    4.594 0.000 0.000
    0.000 4.594 0.000
    2.959 0.000 4.594
   Ti   O
    2    4
Line
   0.0000000  0.0000000  0.0000000  0.5000000  0.5000000  0.5000000
   0.3053003  0.3053003  0.0000000  0.1946997  0.8053003  0.5000000
   0.6946997  0.6946997  0.0000000  0.3053003  0.1946997  0.5000000
   0.1946997  0.8053003  0.5000000  0.6946997  0.6946997  0.0000000
   0.8053003  0.1946997  0.5000000  0.8053003  0.1946997  0.0000000
   0.0000000  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000
```

> 注: VASP 6.x支持LINEAR_COUPLING自动在初始和终态间插值

### 2.2 离散λ点计算

更可靠的方法是手动设置多个λ点：

```bash
# 创建λ点目录
for lambda in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    mkdir lambda_$lambda
done
```

**run_ti.sh**:
```bash
#!/bin/bash
# 热力学积分工作流

LAMBDAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
TEMPS=(300 300 300 300 300 300 300 300 300 300 300)

for i in {0..10}; do
    LAMBDA=${LAMBDAS[$i]}
    TEMP=${TEMPS[$i]}
    
    mkdir -p lambda_$LAMBDA
    cd lambda_$LAMBDA
    
    # 生成当前λ点的结构
    python ../interpolate_structure.py $LAMBDA ../initial.vasp ../final.vasp > POSCAR
    
    # 写入INCAR
    cat > INCAR << EOF
SYSTEM = TI lambda=$LAMBDA
ENCUT = 400
ISMEAR = 0
SIGMA = 0.1

# MD设置
MDALGO = 2
TEBEG = $TEMP
TEEND = $TEMP
POTIM = 1.0
NSW = 10000

# 输出控制
LCHARG = .FALSE.
LWAVE = .FALSE.
EOF
    
    # 运行
    cp ../KPOINTS ../POTCAR .
    mpirun -np 16 vasp_std > vasp.out
    
    # 提取dH/dλ
    grep 'dE_dlambda' OSZICAR | tail -1 >> ../dhdl.dat
    
    cd ..
done
```

**interpolate_structure.py**:
```python
#!/usr/bin/env python3
"""结构插值工具"""
import sys
import numpy as np
from ase.io import read, write
from ase.geometry import find_mic

lambda_val = float(sys.argv[1])
initial = read(sys.argv[2])
final = read(sys.argv[3])

# 插值
# 处理周期性边界条件
cell = initial.get_cell()
pos_i = initial.get_scaled_positions()
pos_f = final.get_scaled_positions()

# 最小镜像约定
diff = pos_f - pos_i
diff -= np.round(diff)
pos_interp = pos_i + lambda_val * diff

# 创建新结构
interp = initial.copy()
interp.set_scaled_positions(pos_interp)

# 输出VASP格式
print("Interpolated Structure")
write('-', interp, format='vasp')
```

### 2.3 结果分析

**extract_fe.py**:
```python
#!/usr/bin/env python3
"""提取并积分自由能"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# 读取dH/dλ数据
# 格式: lambda, dH/dλ (eV), std_error
data = np.loadtxt('dhdl.dat')
lambdas = data[:, 0]
dhdl_mean = data[:, 1]
dhdl_std = data[:, 2]

# 数值积分 (Simpson法则)
delta_f = simpson(dhdl_mean, lambdas)

# 误差传播 (假设各λ点独立)
error = np.sqrt(np.sum((dhdl_std * np.gradient(lambdas))**2))

print(f"ΔF = {delta_f:.4f} ± {error:.4f} eV")
print(f"ΔF = {delta_f * 23.06:.2f} ± {error * 23.06:.2f} kcal/mol")

# 绘制
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(lambdas, dhdl_mean, yerr=dhdl_std, fmt='o-', capsize=4)
ax.fill_between(lambdas, dhdl_mean - dhdl_std, dhdl_mean + dhdl_std, alpha=0.3)
ax.set_xlabel('λ (Reaction Coordinate)')
ax.set_ylabel(r'$\langle \partial H/\partial \lambda \rangle$ (eV)')
ax.set_title(f'Thermodynamic Integration: ΔF = {delta_f:.3f} ± {error:.3f} eV')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ti_curve.png', dpi=150)
```

---

## 3. Quantum ESPRESSO + CPMD 自由能

### 3.1 CP分子动力学

```bash
mkdir qe_fe && cd qe_fe
```

**cp.in** (Car-Parrinello MD):
```fortran
&CONTROL
    calculation = 'cp'
    restart_mode = 'from_scratch'
    prefix = 'water_fe'
    outdir = './tmp/'
    pseudo_dir = '../pseudo/'
    nstep = 5000
    dt = 5.0d0          ! 时间步长 (a.u.)
    iprint = 100
    isave = 1000
    tstress = .true.
    tprnfor = .true.
/
&SYSTEM
    ibrav = 1
    celldm(1) = 20.0    ! 20 a.u. ~ 10.58 Angstrom
    nat = 3
    ntyp = 2
    ecutwfc = 30
    ecutrho = 240
    nosym = .true.
    occupations = 'smearing'
    smearing = 'gaussian'
    degauss = 0.01
/
&ELECTRONS
    electron_dynamics = 'damp'
    electron_damping = 0.2
    emass = 400.d0
/
&IONS
    ion_dynamics = 'verlet'
    ion_temperature = 'nose'
    tempw = 300.0       ! 目标温度 (K)
    fnosep = 4.0        ! Nose频率
/
&CELL
    cell_dynamics = 'none'
/
ATOMIC_SPECIES
 O  15.999 O.pbe-n-kjpaw_psl.1.0.0.UPF
 H  1.008  H.pbe-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (bohr)
 O  0.0  0.0  0.0
 H  1.4  1.4  0.0
 H -1.4  1.4  0.0
```

### 3.2 TI-DPS (Thermodynamic Integration with Dynamical Path Sampling)

```bash
# 使用PLUMED进行增强采样+TI
```

**plumed.dat** (炼金变换):
```
# 水分子解离自由能计算
# 反应坐标: O-H键长

# 定义集体变量
dist: DISTANCE ATOMS=1,2

# 炼金变换: H2O -> H2 + O (理论示例)
# 实际: 改变势能面

# 偏置势 (TI)
RESTRAINT ARG=dist AT=1.0 KAPPA=100.0

# 打印
dist: DISTANCE ATOMS=1,2
PRINT ARG=dist STRIDE=100 FILE=COLVAR
```

### 3.3 alchemical-transformation.py

使用pymbar进行MBAR分析:

```python
#!/usr/bin/env python3
"""
炼金自由能计算 (Alchemical Free Energy)
使用MBAR (Multistate Bennett Acceptance Ratio)
"""
import numpy as np
import pymbar
import matplotlib.pyplot as plt

# λ点和对应的数据文件
lambdas = np.linspace(0, 1, 11)
n_states = len(lambdas)

# 读取每个λ点的能量数据
u_kln = np.zeros([n_states, n_states, 5000])  # [k状态, l状态, 样本]
N_k = np.zeros(n_states)

for k, lam in enumerate(lambdas):
    # 读取当前λ点的能量
    # 格式: step, potential_energy_kinetic_energy, ...
    data = np.loadtxt(f'lambda_{lam:.1f}/energies.dat')
    u_kln[k, k, :len(data)] = data[:, 1]  # 势能
    N_k[k] = len(data)

# 使用MBAR计算自由能
mbar = pymbar.MBAR(u_kln, N_k)

# 提取自由能差
results = mbar.getFreeEnergyDifferences()
delta_f = results['Delta_f']
delta_f_std = results['dDelta_f']

print("Free Energy Differences (kT units):")
for i in range(n_states - 1):
    print(f"λ={lambdas[i]:.1f} -> λ={lambdas[i+1]:.1f}: "
          f"{delta_f[i, i+1]:.4f} ± {delta_f_std[i, i+1]:.4f}")

total_dg = delta_f[0, -1]
print(f"\nTotal ΔG = {total_dg:.4f} kT = {total_dg * 0.592:.3f} kcal/mol (at 300K)")

# 绘制PMF
pmf_results = mbar.computePMF(u_kln[:, 0, :], lambdas, n_bins=50)
plt.plot(pmf_results['bin_centers'], pmf_results['f_i'])
plt.xlabel('Reaction Coordinate λ')
plt.ylabel('Free Energy (kT)')
plt.savefig('pmf_alchemical.png', dpi=150)
```

---

## 4. 增强采样结合自由能计算

### 4.1 Metadynamics + TI

```bash
# 使用PLUMED进行well-tempered metadynamics
```

**plumed_metad.dat**:
```
# 定义CV
dist: DISTANCE ATOMS=1,2
angle: ANGLE ATOMS=3,1,2

# Metadynamics
metad: METAD ARG=dist,angle SIGMA=0.1,0.2 HEIGHT=1.2 BIASFACTOR=10 PACE=500 FILE=HILLS

# 计算加权自由能 (reweighting)
# 使用 umbrella integration

PRINT ARG=dist,angle,metad.bias STRIDE=100 FILE=COLVAR
```

**reweighting.py**:
```python
#!/usr/bin/env python3
"""
Metadynamics reweighting for free energy
"""
import numpy as np
import matplotlib.pyplot as plt

# 读取COLVAR
data = np.loadtxt('COLVAR', comments='#')
dist = data[:, 1]
angle = data[:, 2]
bias = data[:, 3]  # metadynamics偏置势

# 计算权重 (接受-拒绝)
beta = 1.0 / (8.617e-5 * 300)  # 1/eV at 300K
weights = np.exp(beta * bias)
weights /= np.sum(weights)

# 2D自由能面
H, xedges, yedges = np.histogram2d(dist, angle, bins=50, weights=weights)
F = -np.log(H.T + 1e-10) / beta
F -= F.min()

# 绘制
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
plt.contourf(X, Y, F, levels=20, cmap='viridis')
plt.colorbar(label='Free Energy (eV)')
plt.xlabel('Distance (Å)')
plt.ylabel('Angle (rad)')
plt.savefig('fes_2d.png', dpi=150)
```

### 4.2 Umbrella Sampling + WHAM

```bash
# 创建窗口
for window in {0..20}; do
    mkdir window_$window
    # 在每个窗口设置不同的约束位置
    # 使用PLUMED的RESTRAINT
    cp template/* window_$window/
    sed -i "s/AT=0.5/AT=$(echo "scale=2; $window * 0.1" | bc)/" window_$window/plumed.dat
done
```

**wham_analysis.py**:
```python
#!/usr/bin/env python3
"""
Weighted Histogram Analysis Method (WHAM)
"""
import numpy as np
from pymbar import timeseries

# 读取各窗口数据
windows = 21
cv_min, cv_max = 0.0, 2.0
k_spring = 500.0  # kcal/(mol*A^2)

cv_data = []
bias_centers = np.linspace(cv_min, cv_max, windows)

for i in range(windows):
    data = np.loadtxt(f'window_{i}/COLVAR')[:, 1]
    cv_data.append(data)

# WHAM迭代求解
N = [len(d) for d in cv_data]
total_N = sum(N)

# 初始猜测
F = np.zeros(windows)

# 迭代
for iteration in range(1000):
    F_old = F.copy()
    
    # 计算权重因子
    denom = np.zeros(total_N)
    idx = 0
    for i in range(windows):
        for j in range(N[i]):
            # 该样本在所有窗口的偏置势能
            bias_all = 0.5 * k_spring * (cv_data[i][j] - bias_centers)**2
            denom[idx] = np.sum(N * np.exp(-bias_all + F))
            idx += 1
    
    # 更新自由能
    for i in range(windows):
        numerator = np.sum([np.exp(-0.5 * k_spring * (cv - bias_centers[i])**2) 
                           for cv in cv_data[i]])
        F[i] = -np.log(numerator / np.mean(1.0 / denom))
    
    # 归一化
    F -= F[0]
    
    if np.max(np.abs(F - F_old)) < 1e-6:
        print(f"WHAM converged at iteration {iteration}")
        break

# 输出PMF
cv_fine = np.linspace(cv_min, cv_max, 100)
pmf = np.interp(cv_fine, bias_centers, F)

np.savetxt('pmf_wham.dat', np.column_stack([cv_fine, pmf]))
```

---

## 5. 绝对自由能计算

### 5.1 爱因斯坦晶体法

用于计算固体的绝对自由能：

```python
#!/usr/bin/env python3
"""
Einstein Crystal Method for Absolute Free Energy
"""
import numpy as np
from scipy.integrate import quad

# 参数
T = 300  # K
k_B = 8.617e-5  # eV/K
beta = 1.0 / (k_B * T)
hbar = 6.582e-16  # eV*s

# Einstein频率 (从声子DOS获得)
omega_E = 10.0  # THz ~ 10^13 Hz
omega_E *= 2 * np.pi * 1e12  # 转为 rad/s

# 爱因斯坦晶体自由能 (参考态)
F_Einstein = 3 * k_B * T * np.log(beta * hbar * omega_E)

# 从爱因斯坦晶体到实际晶体的FEP积分
# 计算 ⟨U_real - U_Einstein⟩_λ

def integrand(lambda_val):
    """计算给定λ的dF/dλ"""
    # 从模拟数据插值
    # 这需要实际MD数据
    return lambda_val * 0.1  # 占位符

# 数值积分
F_correction, _ = quad(integrand, 0, 1)

F_total = F_Einstein + F_correction
print(f"Einstein crystal F = {F_Einstein:.4f} eV/atom")
print(f"Correction = {F_correction:.4f} eV/atom")
print(f"Total F = {F_total:.4f} eV/atom")
```

### 5.2 Frenkel-Ladd方法

```python
#!/usr/bin/env python3
"""
Frenkel-Ladd method for absolute free energy
"""
import numpy as np

# 将实际系统与谐振子参考态耦合
# H(λ) = (1-λ) H_real + λ H_harmonic

# 谐振子参考态的自由能 (可解析计算)
def F_harmonic(k_spring, T):
    """3D谐振子的自由能"""
    beta = 1.0 / (8.617e-5 * T)
    hbar_omega = np.sqrt(k_spring / m) * hbar  # 需要质量m
    return 3 * k_B * T * np.log(2 * np.sinh(beta * hbar_omega / 2))

# TI计算
lambdas = np.linspace(0, 1, 11)
U_diff_mean = np.zeros(len(lambdas))

for i, lam in enumerate(lambdas):
    # 运行该λ点的MD
    # 计算 ⟨U_harmonic - U_real⟩_λ
    U_diff_mean[i] = run_md_and_compute(lam)

# 积分
from scipy.integrate import simpson
F_abs = F_harmonic(k_ref, T) - simpson(U_diff_mean, lambdas)
```

---

## 6. 相图构建 (固-液-气)

### 6.1 共存法

```python
#!/usr/bin/env python3
"""
Phase diagram construction from free energy
"""
import numpy as np
import matplotlib.pyplot as plt

# 温度范围
T_range = np.linspace(100, 2000, 100)

# 各相的自由能曲线 (从TI计算获得)
F_solid = compute_f_solid(T_range)  # 爱因斯坦晶体法
F_liquid = compute_f_liquid(T_range)  # 炼金变换
F_gas = compute_f_gas(T_range)  # 理想气体近似

# 找交点 (相变点)
def find_intersection(T, F1, F2):
    """找到F1=F2的温度"""
    diff = np.abs(F1 - F2)
    idx = np.argmin(diff)
    return T[idx]

T_melt = find_intersection(T_range, F_solid, F_liquid)
T_boil = find_intersection(T_range, F_liquid, F_gas)

print(f"Melting point: {T_melt:.1f} K")
print(f"Boiling point: {T_boil:.1f} K")

# 绘制相图
plt.plot(T_range, F_solid, 'b-', label='Solid')
plt.plot(T_range, F_liquid, 'r-', label='Liquid')
plt.plot(T_range, F_gas, 'g-', label='Gas')
plt.axvline(T_melt, color='k', linestyle='--', label=f'Tm={T_melt:.0f}K')
plt.axvline(T_boil, color='gray', linestyle='--', label=f'Tb={T_boil:.0f}K')
plt.xlabel('Temperature (K)')
plt.ylabel('Free Energy (eV/atom)')
plt.legend()
plt.savefig('phase_diagram.png', dpi=150)
```

---

## 7. 故障排查

### 7.1 TI滞回问题

**现象**: 正向和反向积分结果不一致

**解决方案**:
```python
# 检查hysteresis
forward = np.loadtxt('ti_forward.dat')
backward = np.loadtxt('ti_backward.dat')

# 应该对称
assert np.allclose(forward[::-1], backward, atol=0.01), "Hysteresis detected!"

# 对策:
# 1. 增加每个λ点的采样时间
# 2. 使用更小的λ步长
# 3. 检查结构变换是否连续
```

### 7.2 端点奇异性

**现象**: λ接近0或1时dH/dλ发散

**软核势**:
```python
# 使用软核势避免奇异性
def soft_core(lambda_val, alpha=0.5):
    """软核变换"""
    return lambda_val**alpha

# 修改耦合方式
lambda_eff = soft_core(lambda_val)
```

### 7.3 统计误差控制

```python
# 分块分析
from pymbar import timeseries

# 检测相关时间
t_data = np.loadtxt('trajectory.dat')[:, 1]
t_equil, g, Neff_max = timeseries.detectEquilibration(t_data)

# 只使用平衡后的数据
t_equilibrated = t_data[t_equil:]

# 有效独立样本数
N_eff = len(t_equilibrated) / g
print(f"Correlation time: {g:.1f} steps")
print(f"Effective samples: {N_eff:.0f}")
```

---

## 8. 推荐工作流

### 8.1 标准TI工作流

```bash
#!/bin/bash
# standard_ti_workflow.sh

SYSTEM="SrTiO3"
LAMBDAS=$(seq 0 0.1 1)

# 步骤1: 准备初始和终态结构
prepare_endpoints.py --initial phase_A.cif --final phase_B.cif

# 步骤2: 运行各λ点的平衡MD
for lam in $LAMBDAS; do
    mkdir lambda_$lam && cd lambda_$lam
    
    # 结构插值
    interpolate.py $lam ../phase_A.vasp ../phase_B.vasp > POSCAR
    
    # 短平衡
    run_vasp.sh --steps 5000 --temperature 300
    
    # 生产运行
    run_vasp.sh --steps 50000 --restart
    
    # 提取dH/dλ
    extract_dhdl.py OSZICAR >> ../dhdl.dat
    
    cd ..
done

# 步骤3: 分析
analyze_ti.py dhdl.dat --output free_energy.png
```

### 8.2 快速TI (Slo-Gro)

```bash
# 单轨迹慢增长 (快速估计)
vasp_std > vasp.out  # LINTERFAST=.TRUE.

# 提取即时自由能
grep 'free energy' OSZICAR > fe_trajectory.dat
```

---

## 9. 参考资源

### 9.1 关键文献

1. **Frenkel & Smit** (2002). Understanding Molecular Simulation. *Academic Press*.
2. **Tuckerman** (2010). Statistical Mechanics: Theory and Molecular Simulation. *Oxford*.
3. **Shirts & Chodera** (2008). Statistically optimal analysis of samples from multiple equilibrium states. *J. Chem. Phys.* 129, 124105.
4. **Kästner & Thiel** (2005). Bridging the gap between thermodynamic integration and umbrella sampling. *J. Chem. Phys.* 123, 144104.

### 9.2 软件工具

- **pymbar**: https://github.com/choderalab/pymbar
- **alchemical-analysis**: https://github.com/MobleyLab/alchemical-analysis
- **PLUMED**: https://www.plumed.org/
- **SSAGES**: https://ssagesproject.github.io/

### 9.3 最佳实践

1. 始终检查hysteresis
2. 使用MBAR/BAR而非简单TI提高精度
3. 足够的平衡时间 (至少10ps)
4. 足够的生产时间 (至少100ps每λ点)
5. 保存所有原始能量数据便于重新分析
