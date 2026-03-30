# DMFT 动态平均场理论计算指南

## 1. 理论基础

### 1.1 DMFT简介
动态平均场理论(Dynamical Mean-Field Theory, DMFT)将强关联电子体系的晶格问题映射到有效杂质问题上，能够正确处理电子关联效应。

**核心思想**:
- 忽略空间涨落，保留时间涨落
- 自洽求解杂质问题和晶格问题
- Hubbard模型和Anderson杂质模型的对应

**适用体系**:
- 过渡金属氧化物 (Mott绝缘体)
- f电子体系 (重费米子)
- 高温超导材料
- 强关联拓扑材料

### 1.2 DFT+DMFT框架

```
┌─────────────────────────────────────────────┐
│           DFT+DMFT 自洽循环                   │
├─────────────────────────────────────────────┤
│  DFT计算 → 构建Wannier轨道 → 投影Hubbard模型  │
│       ↑                                    │
│  更新电荷密度 ← 计算DMFT格林函数 ← 杂质求解器  │
└─────────────────────────────────────────────┘
```

**主要代码**:
- **Questaal (lmf-Mark)** - 全势LMTO+DMFT实现
- **Wien2K + TRIQS/EDMFT** - FLAPW+连续时间量子蒙特卡洛(CTQMC)
- **VASP + VASPKIT + TRIQS** - PAW+DMFT
- **Quantum ESPRESSO + Wannier90 + TRIQS** - 平面波+DMFT
- **ABINIT + TRIQS** - 内置DMFT功能

---

## 2. Questaal DFT+DMFT 计算

### 2.1 环境准备

```bash
# 安装Questaal
wget https://www.questaal.org/downloads/questaal.tar.gz
tar -xzf questaal.tar.gz
cd questaal
cp site.site.gcc site.site
make all

# 设置环境变量
export PATH=$PATH:/path/to/questaal/bin
export OMP_NUM_THREADS=4  # 并行线程
```

### 2.2 初始化计算 (SrVO3示例)

```bash
# 创建工作目录
mkdir srvo3_dmft && cd srvo3_dmft

# 获取结构文件
blm srvo3 --ctrl --wsit > blm.log

# 编辑srvo3.sites
# SrVO3: 钙钛矿结构，空间群Pm-3m
```

**srvo3.site**:
```
%SITE
  ATOM=Sr POS=0.0000000 0.0000000 0.0000000
  ATOM=V  POS=0.5000000 0.5000000 0.5000000
  ATOM=O  POS=0.5000000 0.5000000 0.0000000
  ATOM=O  POS=0.5000000 0.0000000 0.5000000
  ATOM=O  POS=0.0000000 0.5000000 0.5000000
```

### 2.3 DFT基态计算

```bash
# 生成控制文件
blm srvo3 --nfile --rdsite=srvo3.site --express=0 --mag --gw --dmft > blm.log

# 编辑ctrl.srvo3，设置关键参数
```

**ctrl.srvo3 关键参数**:
```
# 基本DFT设置
HAM    NSPIN=1         # 非磁性
       GMAX=12         # 平面波截断
       FTMESH=30       # FFT网格

# DMFT设置 (关键)
DMFT   NOMEGA=30       # Matsubara频率数
       BETA=50         # 逆温度 (eV^-1), T=232K
       LMAXW=2         # 最大角动量
       MULL=2          # Mulliken投影

#  Hubbard U参数
H      UJ=4.0,0.3     # U=4.0 eV, J=0.3 eV (V-3d)
```

```bash
# 自洽计算
lmf srvo3 -vnit=30 --rs=1 --pr31 > lmf.log

# 检查收敛
grep 'h ns' lmf.log | tail -5
```

### 2.4 DMFT自洽循环

```bash
# 1. 准备DMFT输入
lmgw srvo3 --dmft --job=-1 > dmft_init.log

# 2. 运行DMFT自洽循环
for iter in {1..20}; do
    echo "=== DMFT Iteration $iter ==="
    
    # DMFT求解器 (CTQMC)
    lmgw srvo3 --dmft --job=1 > dmft_iter${iter}.log
    
    # 检查收敛
    grep 'Delta Q' dmft_iter${iter}.log
    
    # 提取占据数
    awk '/occ/{print $2, $3}' dmft_iter${iter}.log > occ_iter${iter}.dat
done
```

### 2.5 后处理与分析

```bash
# 提取自能
lmf srvo3 --rs=1 --pr41 > sigma.log
cp sig.srvo3 sig.dat

# 绘制Matsubara频率上的自能
python << 'EOF'
import numpy as np
import matplotlib.pyplot as plt

# 读取自能数据
data = np.loadtxt('sig.dat', skiprows=1)
w_n = data[:, 0]  # Matsubara频率
sigma_real = data[:, 1]
sigma_imag = data[:, 2]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# 实部
ax[0].plot(w_n, sigma_real, 'bo-', markersize=4)
ax[0].set_xlabel(r'$i\omega_n$ (eV)')
ax[0].set_ylabel(r'Re $\Sigma(i\omega_n)$ (eV)')
ax[0].set_title('Self-energy Real Part')

# 虚部
ax[1].plot(w_n, -sigma_imag, 'ro-', markersize=4)
ax[1].set_xlabel(r'$i\omega_n$ (eV)')
ax[1].set_ylabel(r'-Im $\Sigma(i\omega_n)$ (eV)')
ax[1].set_title('Self-energy Imaginary Part')

plt.tight_layout()
plt.savefig('self_energy.png', dpi=150)
plt.close()
EOF

# 计算谱函数 (需要最大熵法或Pade近似)
lmf srvo3 --rs=1 --band~fn=syml > band_dmft.log
```

---

## 3. VASP + VASPKIT + TRIQS 计算

### 3.1 准备工作流

```bash
# 步骤1: VASP标准DFT计算
mkdir 1_dft && cd 1_dft
```

**INCAR** (标准DFT):
```
SYSTEM = SrVO3 PBE0+DMFT

# 基础设置
ENCUT = 500
EDIFF = 1E-7
ISMEAR = 0
SIGMA = 0.05

# 电子步
ALGO = Normal
NELM = 100
NELMIN = 6

# 结构优化 (可选)
ISIF = 3
NSW = 50
EDIFFG = -0.01

# Wannier投影
LWAVE = .TRUE.
LCHARG = .TRUE.
```

```bash
# 运行VASP
mpirun -np 16 vasp_std

# 回到上级目录
cd ..
```

### 3.2 Wannier函数投影

```bash
# 步骤2: Wannier90计算
mkdir 2_wannier && cd 2_wannier
cp ../1_dft/CHGCAR ../1_dft/WAVECAR ../1_dft/POSCAR .
```

**INCAR.wannier**:
```
SYSTEM = SrVO3 Wannier
ENCUT = 500
ISMEAR = 0
SIGMA = 0.05

# Wannier90接口
LWANNIER90 = .TRUE.
LWRITE_MMN_AMN = .TRUE.

# 投影设置 (V-3d轨道)
NUM_WANN = 3
PROJECTIONS = V:d
```

**wannier90.win**:
```
num_wann = 3
num_iter = 1000

# 投影
begin projections
V: dxy;dxz;dyz
end projections

# k点网格
mp_grid : 6 6 6

begin kpoints
# 自动生成的k点
end kpoints

# 离壳能量窗口
dis_win_min = -2.0
dis_win_max = 8.0
dis_froz_min = -1.0
dis_froz_max = 3.0

# 输出
write_hr = .true.
write_xyz = .true.
bands_plot = .true.
```

```bash
# 运行VASP+Wannier90
vasp_std > vasp.out
wannier90.x wannier90

cd ..
```

### 3.3 DFT+DMFT 计算 (TRIQS)

```bash
# 步骤3: DMFT计算
mkdir 3_dmft && cd 3_dmft

# 安装TRIQS (如未安装)
# pip install triqs triqs_cthyb
```

**dmft_srvo3.py** (TRIQS脚本):
```python
#!/usr/bin/env python3
"""
SrVO3 DFT+DMFT calculation using TRIQS
"""

from triqs.gf import *
from triqs.operators import *
from triqs_cthyb import Solver
from h5 import HDFArchive
import numpy as np

# 参数设置
beta = 50.0          # 逆温度 (1/eV)
U = 4.0              # Hubbard U (eV)
J = 0.3              # Hund's coupling (eV)
n_iw = 1024          # Matsubara频率数
n_tau = 10001        # 虚时间格点数
n_cycles = 100000    # QMC循环数

# 读取Wannier哈密顿量
H_k = np.load('../2_wannier/hr.dat.npy')  # 需要转换格式

# 初始化求解器
S = Solver(beta=beta, gf_struct=[('up', 3), ('dn', 3)], n_tau=n_tau, n_iw=n_iw)

# 构建非相互作用格林函数
G0_iw = S.G0_iw

# DMFT自洽循环
for iter_num in range(30):
    print(f"\n=== DMFT Iteration {iter_num + 1} ===")
    
    # 计算晶格格林函数 (通过Dyson方程)
    # G^{-1} = G_0^{-1} - Sigma
    
    # 计算 Weiss场
    # G_0^{-1} = (sum_k G(k))^{-1} + Sigma
    
    # 设置CTQMC参数
    ops = c_dag('up', 0) * c('up', 0)  # 密度算符示例
    
    # 运行CTQMC求解器
    S.solve(
        h_int=U * n('up', 0) * n('dn', 0),  # Hubbard相互作用
        n_cycles=n_cycles,
        length_cycle=200,
        n_warmup_cycles=5000,
        measure_G_tau=True,
        measure_G_l=False
    )
    
    # 提取自能
    Sigma_iw = S.Sigma_iw
    
    # 检查收敛
    diff = np.max(np.abs(Sigma_iw['up'].data - Sigma_iw_prev['up'].data)) if iter_num > 0 else 1.0
    print(f"Self-energy difference: {diff:.6f}")
    
    if diff < 1e-4:
        print("DMFT converged!")
        break
    
    Sigma_iw_prev = Sigma_iw.copy()

# 保存结果
with HDFArchive('dmft_results.h5', 'w') as ar:
    ar['Sigma_iw'] = Sigma_iw
    ar['G_iw'] = S.G_iw
    ar['G_tau'] = S.G_tau

print("DMFT calculation completed!")
```

```bash
# 运行DMFT计算
python dmft_srvo3.py > dmft.log 2>&1
```

### 3.4 使用VASPKIT自动DMFT

```bash
# VASPKIT v1.4+ 支持DMFT接口
vaspkit -task 802   # DFT+DMFT准备

# 生成的文件:
# - wannier90_hr.dat (紧束缚哈密顿量)
# - wannier90_centres.xyz (Wannier中心)
# - INCAR_DMFT (DMFT输入模板)
```

---

## 4. Quantum ESPRESSO + TRIQS 计算

### 4.1 DFT计算

```bash
mkdir qe_dmft && cd qe_dmft
```

**pw.in** (QE输入):
```fortran
&CONTROL
    calculation = 'scf'
    prefix = 'srvo3'
    outdir = './tmp/'
    pseudo_dir = '../pseudo/'
/
&SYSTEM
    ibrav = 1
    A = 3.84
    nat = 5
    ntyp = 3
    ecutwfc = 60
    ecutrho = 480
    occupations = 'smearing'
    smearing = 'gaussian'
    degauss = 0.02
    nspin = 1
    lda_plus_u = .true.
    Hubbard_U(2) = 1.0e-10  ! 初始U值
    Hubbard_J(2) = 0.0
/
&ELECTRONS
    conv_thr = 1.0d-10
/
ATOMIC_SPECIES
 Sr 87.62 Sr.pbe-spn-kjpaw_psl.1.0.0.UPF
 V  50.94 V.pbe-spn-kjpaw_psl.1.0.0.UPF
 O  16.00 O.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
 Sr 0.0 0.0 0.0
 V  0.5 0.5 0.5
 O  0.5 0.5 0.0
 O  0.5 0.0 0.5
 O  0.0 0.5 0.5
K_POINTS (automatic)
 8 8 8 0 0 0
```

```bash
mpirun -np 16 pw.x -in pw.in > pw.out
```

### 4.2 Wannier90 + pw2wannier90

**pwscf2wannier90.in**:
```fortran
&inputpp
    outdir = './tmp/'
    prefix = 'srvo3'
    seedname = 'srvo3'
    write_mmn = .true.
    write_amn = .true.
    write_unk = .true.
/
```

**srvo3.win** (Wannier90输入):
```
num_wann = 3
num_iter = 1000

begin projections
f=0.5,0.5,0.5:l=2  ! V-3d轨道
end projections

bands_plot = true

begin kpoint_path
G 0.0 0.0 0.0  M 0.5 0.5 0.0
M 0.5 0.5 0.0  X 0.5 0.0 0.0
X 0.5 0.0 0.0  G 0.0 0.0 0.0
G 0.0 0.0 0.0  R 0.5 0.5 0.5
end kpoint_path
```

```bash
# 非自洽计算
mpirun -np 16 pw.x -in pw_nscf.in > pw_nscf.out

# 生成重叠矩阵
mpirun -np 16 pw2wannier90.x -in pwscf2wannier90.in > pw2wan.out

# Wannier化
wannier90.x -pp srvo3
wannier90.x srvo3
```

### 4.3 DFT+DMFT (TRIQS/DFTTools)

```bash
# 安装DFTTools
# pip install triqs_dfttools
```

**srvo3_dmft.py**:
```python
#!/usr/bin/env python3
"""
QE+Wannier90+TRIQS DFT+DMFT for SrVO3
"""

from triqs_dfttools.sumk_dft import SumkDFT
from triqs_dfttools.sumk_dft_tools import SumkDFTTools
from triqs.gf import *
from triqs.operators import *
from triqs_cthyb import Solver
import numpy as np

# 初始化Sumk
SK = SumkDFT(hdf_file='srvo3.h5', use_dft_blocks=True)

# 参数
beta = 50.0
U = 4.0
J = 0.3

# 求解器设置
S = Solver(beta=beta, 
           gf_struct=SK.gf_struct_solver[0],
           n_tau=10001,
           n_iw=1024)

# DMFT自洽循环
for iteration in range(30):
    print(f"\n=== Iteration {iteration} ===")
    
    # 计算晶格格林函数
    SK.put_Sigma(Sigma_imp=[S.Sigma_iw])
    G_loc = SK.extract_G_loc()[0]
    
    # 计算Weiss场
    S.G0_iw << inverse(SK.eff_atomic_levels()[0] - SK.hopping[0])
    
    # CTQMC求解
    S.solve(
        h_int=Operator(),  # 由DFTTools自动构建
        n_cycles=50000,
        length_cycle=200
    )
    
    # 计算占据数
    n_imp = sum([g.total_density() for g in S.G_iw.values()])
    print(f"Impurity occupancy: {n_imp:.4f}")

# 分析结果
SK.analyse(block='up', threshold=0.1)
```

---

## 5. 参数选择与收敛测试

### 5.1 Hubbard U值确定

```bash
# 方法1: 线性响应 (cRPA)
# VASP
INCAR_LRESP:
LMODELHF = .TRUE.
LRHFCALC = .TRUE.
ENCUTGW = 250

# 计算得到 U = 3.5-4.5 eV for V-3d
```

```bash
# 方法2: cRPA (Questaal)
lmf srvo3 --crpa > crpa.log
grep 'U=' crpa.log
```

### 5.2 温度/Beta选择

| 温度(K) | Beta (eV^-1) | 适用体系 |
|---------|--------------|----------|
| 116     | 100          | 低温物理 |
| 232     | 50           | 标准计算 |
| 464     | 25           | 高温/金属 |
| 1160    | 10           | 快速测试 |

### 5.3 Matsubara频率收敛

```python
# 检查频率收敛
import numpy as np
import matplotlib.pyplot as plt

for n_iw in [512, 1024, 2048]:
    # 运行DMFT
    # ...
    # 比较自能
    plt.plot(w_n[:100], sigma_imag[:100], label=f'n_iw={n_iw}')

plt.legend()
plt.savefig('convergence_niw.png')
```

---

## 6. 结果分析

### 6.1 谱函数计算

```python
# Pade近似或最大熵法得到实频率谱
from triqs.gf.tools import compute_tail
from triqs_maxent import *

# 最大熵法
bl = BlockGf(name_list=['up', 'dn'], block_list=[G_tau['up'], G_tau['dn']])

for name, g in bl:
    tm = TauMaxEnt(cost_function='bryan')
    tm.set_G_tau(g)
    tm.set_error(1e-4)
    tm.omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=200)
    
    result = tm.run()
    A_w = result.get_A()
    
    # 保存
    np.savetxt(f'spectral_{name}.dat', 
               np.column_stack([tm.omega, A_w]))
```

### 6.2 动量分辨谱函数

```python
# 计算A(k,w)
# A(k,w) = -1/pi * Im G(k,w)
# G(k,w) = [w + mu - e_k - Sigma(w)]^{-1}

# k点路径
k_path = np.linspace(0, 1, 100)

# 计算
A_kw = np.zeros((len(k_path), len(omega)))

for ik, k in enumerate(k_path):
    e_k = interpolate_band(k)  # 从Wannier插值
    for iw, w in enumerate(omega):
        G_kw = 1.0 / (w + 1j*eta + mu - e_k - Sigma_w[iw])
        A_kw[ik, iw] = -1.0/np.pi * np.imag(G_kw)

# 绘制
plt.imshow(A_kw, extent=[omega[0], omega[-1], 0, 1], 
           aspect='auto', origin='lower', cmap='hot')
plt.colorbar(label='A(k,w)')
plt.savefig('akw_spectral.png', dpi=200)
```

### 6.3 物理量提取

```python
# 准粒子重整化因子 Z
# Z = [1 - dReSigma/dw|_{w=0}]^{-1}

# 从自能数据计算
w_small = np.linspace(-0.5, 0.5, 100)
re_sigma = np.interp(w_small, omega, sigma_real)
dsigma_dw = np.gradient(re_sigma, w_small)
Z = 1.0 / (1.0 - dsigma_dw[len(dsigma_dw)//2])

print(f"Quasiparticle weight Z = {Z:.3f}")

# 有效质量
# m*/m = 1/Z
m_star = 1.0 / Z
print(f"Effective mass ratio m*/m = {m_star:.3f}")
```

---

## 7. 常见问题与故障排查

### 7.1 CTQMC噪声问题

**现象**: 自能噪声大，不收敛

**解决方案**:
```python
# 增加循环数
n_cycles = 500000  # 从100k增加到500k

# 使用Legendre表示
measure_G_l = True
```

### 7.2 电荷不守恒

**现象**: 总占据数偏离目标值

**解决方案**:
```python
# 调整化学势
def adjust_mu(target_n, mu_guess, tol=0.01):
    mu = mu_guess
    while True:
        SK.set_mu(mu)
        n = SK.calc_density()
        if abs(n - target_n) < tol:
            break
        mu += 0.1 * (target_n - n)
    return mu
```

### 7.3 金属化/Mott转变

**诊断**:
```python
# 检查Z因子
if Z < 0.1:
    print("Strongly correlated regime (Mott-like)")
elif Z > 0.8:
    print("Weakly correlated regime")
else:
    print("Intermediate correlation")
```

---

## 8. 高级应用

### 8.1 多轨道DMFT

**La2CuO4 (Cu-d^9)**:
```
# 5个Cu-3d轨道
NUM_WANN = 5

# 全相互作用哈密顿量
# H_int = U * sum(n_i_up * n_i_dn) 
#       + (U-2J) * sum_{i<j, sigma} n_i_sigma * n_j_sigma
#       - J * sum_{i!=j} S_i . S_j
```

### 8.2 团簇DMFT (CDMFT)

```python
# 使用2x2团簇
from triqs_cthyb import Solver

# 团簇自能
S_cdmft = Solver(beta=beta,
                 gf_struct=[('up-1', 1), ('up-2', 1), ('up-3', 1), ('up-4', 1),
                           ('dn-1', 1), ('dn-2', 1), ('dn-3', 1), ('dn-4', 1)])
```

### 8.3 非平衡DMFT

**应用**: 超快光谱、激光诱导相变
```python
# 使用TRIQS/NEGF
from triqs_tprf import *

# 时间依赖哈密顿量
H_t = H_0 + V_pump(t)
```

---

## 9. 参考与资源

### 9.1 推荐文献

1. **Georges, A., et al.** (1996). Dynamical mean-field theory of strongly correlated fermion systems. *Rev. Mod. Phys.* 68, 13.
2. **Kotliar, G., et al.** (2006). Electronic structure calculations with dynamical mean-field theory. *Rev. Mod. Phys.* 78, 865.
3. **Held, K.** (2007). Electronic structure calculations using dynamical mean field theory. *Advances in Physics* 56, 829.

### 9.2 关键软件

- **TRIQS**: https://triqs.github.io/
- **Questaal**: https://www.questaal.org/
- **Wien2K**: http://www.wien2k.at/
- **ABINIT**: https://www.abinit.org/

### 9.3 教程资源

- TRIQS tutorials: https://triqs.github.io/tutorials/
- Questaal DMFT tutorial: https://www.questaal.org/tutorials/dmft/
- Summer School lectures: https://www.cond-mat.de/events/correl19/
