# 路径积分分子动力学 (PIMD)

## 简介

路径积分分子动力学(Path Integral Molecular Dynamics, PIMD)是一种模拟核量子效应(Nuclear Quantum Effects, NQE)的重要方法。与经典分子动力学不同，PIMD基于Feynman的路径积分表述，将量子核描述为由多个"珠子"(beads)组成的环聚合物(ring polymer)，从而能够准确描述零点能、量子隧穿和核量子涨落等效应。

## 理论基础

### 路径积分表述

量子配分函数可以表示为：

$$Z = \text{Tr}(e^{-\beta \hat{H}}) = \oint \mathcal{D}[q(\tau)] e^{-\frac{1}{\hbar}\int_0^{\beta\hbar}[\frac{1}{2}m\dot{q}(\tau)^2 + V(q(\tau))]d\tau}$$

其中 $\beta = 1/(k_B T)$，积分遍历所有闭合路径。

### Trotter分解与环聚合物

通过Trotter分解，配分函数可离散化为：

$$Z_P = \left(\frac{mP}{2\pi\beta\hbar^2}\right)^{3NP/2} \int dR_1 \cdots dR_P \exp\left[-\beta \sum_{i=1}^P \left(\frac{1}{2}m\omega_P^2(R_i - R_{i+1})^2 + \frac{V(R_i)}{P}\right)\right]$$

其中：
- $P$: 珠子数 (Trotter数)
- $\omega_P = P/(\beta\hbar)$: 珠子间谐振频率
- $R_{P+1} = R_1$: 周期性边界条件

**物理图像**: 每个量子粒子被表示为$P$个经典珠子组成的环，珠子间通过谐振弹簧连接。

### 核量子效应的重要性

PIMD能够捕捉以下核量子效应：

1. **零点能效应**: 轻原子(H, D, He)即使在0K也有显著运动
2. **量子隧穿**: 粒子穿过势垒的量子行为
3. **同位素效应**: H/D质量差异导致的性质变化
4. **质子转移**: 氢键网络中的质子运动

---

## PIMD软件包

### i-PI (推荐)

i-PI是一个Python编写的PIMD接口程序，可与多种第一性原理代码配合：

**支持的客户端**:
- Quantum ESPRESSO
- CP2K
- LAMMPS
- VASP (通过外部接口)

**安装**:
```bash
git clone https://github.com/i-pi/i-pi.git
cd i-pi
pip install .
```

**基本输入文件** (`input.xml`):
```xml
<simulation verbosity='medium'>
  <output prefix='simulation'>
    <properties stride='10' filename='out'>[ step, time, conserved, temperature, kinetic_cv, potential ]</properties>
    <trajectory stride='100' filename='pos' cell_units='angstrom'>positions</trajectory>
  </output>
  
  <total_steps>10000</total_steps>
  <prng><seed>12345</seed></prng>
  
  <ffsocket name='qe' mode='inet'>
    <address>localhost</address>
    <port>12345</port>
  </ffsocket>
  
  <system>
    <initialize nbeads='32'>
      <file mode='xyz'>init.xyz</file>
    </initialize>
    
    <forces>
      <force forcefield='qe'/>
    </forces>
    
    <ensemble>
      <temperature units='kelvin'>300</temperature>
    </ensemble>
    
    <motion mode='dynamics'>
      <dynamics mode='nvt'>
        <timestep units='femtosecond'>0.5</timestep>
        <thermostat mode='pile_l'>
          <tau units='femtosecond'>100</tau>
        </thermostat>
      </dynamics>
    </motion>
  </system>
</simulation>
```

**运行**:
```bash
# 终端1: 启动i-PI
i-pi input.xml &

# 终端2: 启动客户端 (如QE)
mpirun -np 4 pw.x --ipi localhost:12345 -in qe.in
```

### PIMD (JAEA)

日本原子力机构开发的PIMD程序，支持多种第一性原理代码接口：

**支持接口**:
- VASP 5.3.5 / 6.4.0
- Quantum ESPRESSO 6.2.1 / 6.3
- CP2K
- GAUSSIAN
- TURBOMOLE

**安装** (以QE接口为例):
```bash
cd pimd
mkdir build && cd build
cmake -DMKLUSE=ON -DQE=ON -DQEVERSION=6.3 ..
make -j 10
```

**输入文件** (`input.dat`):
```
<method>
PIMD

<ensemble>
NVT

<nbead>
32

<dt>
0.25d0

<nstep>
10000

<temperature>
300.d0

<bath_type>
MNHC

<nnhc>
4

<ipotential>
QE

<qe_input_file_name>
pw.in

<qe_output>
1 10
```

### ABINIT内置PIMD

ABINIT从v7.8.2开始内置PIMD功能：

```fortran
# ABINIT输入文件
imgmov 9           # PIMD算法
ntimimage 1000     # MD步数
pitransform 1      # 坐标变换: 0=原始, 1=staging, 2=简正模式
pimass 1.0 1.0     # 虚拟质量 (amu)
mdtemp 300 300     # 初始和恒温器温度
```

---

## 从头算PIMD工作流

### 使用i-PI + Quantum ESPRESSO

#### Step 1: 准备初始结构

```bash
# 创建初始结构文件 init.xyz
# 格式: 第一行原子数，第二行注释，后面是坐标

# 例如: 32个水分子
96
Water 32 molecules
O  0.000  0.000  0.000
H  0.757  0.586  0.000
H -0.757  0.586  0.000
...
```

#### Step 2: i-PI输入文件

```xml
<simulation verbosity='medium'>
  <output prefix='water_pimd'>
    <properties stride='10' filename='out'>
      [ step, time, conserved, temperature{kelvin}, 
        kinetic_cv, potential, pressure_cv{bar} ]
    </properties>
    <trajectory stride='50' filename='xc' format='xyz' cell_units='angstrom'>
      x_centroid
    </trajectory>
    <checkpoint stride='1000'/>
  </output>
  
  <total_steps>50000</total_steps>
  <prng><seed>12345</seed></prng>
  
  <!-- 与QE通信的socket -->
  <ffsocket name='qe' mode='unix'>
    <address>water_pimd</address>
  </ffsocket>
  
  <system>
    <!-- 初始化: 32个珠子 -->
    <initialize nbeads='32'>
      <file mode='xyz'>init.xyz</file>
    </initialize>
    
    <forces>
      <force forcefield='qe'/>
    </forces>
    
    <ensemble>
      <temperature units='kelvin'>300</temperature>
    </ensemble>
    
    <!-- NVT动力学 -->
    <motion mode='dynamics'>
      <dynamics mode='nvt'>
        <timestep units='femtosecond'>0.5</timestep>
        <thermostat mode='pile_l'>
          <tau units='femtosecond'>100</tau>
        </thermostat>
      </dynamics>
    </motion>
  </system>
</simulation>
```

#### Step 3: QE输入文件

```fortran
&CONTROL
  calculation = 'scf'
  prefix = 'water'
  outdir = './tmp/'
  pseudo_dir = '../pseudos/'
  tprnfor = .true.
  tstress = .true.
/

&SYSTEM
  ibrav = 1
  celldm(1) = 18.0
  nat = 96
  ntyp = 2
  ecutwfc = 80
  ecutrho = 400
  occupations = 'smearing'
  smearing = 'mv'
  degauss = 0.01
/

&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.3
/

ATOMIC_SPECIES
O  15.999  O.pbe-n-kjpaw_psl.1.0.0.UPF
H  1.008   H.pbe-kjpaw_psl.1.0.0.UPF

# 坐标将由i-PI通过socket传递
```

#### Step 4: 运行脚本

```bash
#!/bin/bash
# run_pimd.sh

# 启动i-PI
i-pi input.xml &
cd qe_run

# 为每个珠子启动一个QE实例
for i in $(seq 1 32); do
  mkdir -p bead_$i
  cd bead_$i
  mpirun -np 4 pw.x --ipi ../water_pimd --unix -in ../pw.in > pw.out &
  cd ..
done

wait
```

### 使用PIMD + VASP

#### VASP输入准备

**INCAR**:
```
# 基本设置
PREC = Accurate
ENCUT = 600
EDIFF = 1E-8
ISMEAR = 0
SIGMA = 0.05

# MD设置 (PIMD通过接口调用)
IBRION = 1
NSW = 100
ISIF = 3
ISYM = 0

# 输出控制
LCHARG = .FALSE.
LWAVE = .FALSE.
```

**PIMD输入** (`input.dat`):
```
<method>
PIMD

<ensemble>
NPT

<npt_type>
CUBIC2

<nbead>
32

<dt>
0.25d0

<nstep>
10000

<temperature>
300.d0

<pressure>
1.0d0

<bath_type>
MNHC

<nnhc>
4

<ipotential>
VASP6

<vasp_output>
1 10

<vasp_reuse_wavefunction>
1
```

---

## 关键参数设置

### 珠子数 (nbeads/P)

珠子数决定核量子效应的精度：

$$P > \frac{\hbar\omega_{max}}{k_B T}$$

其中 $\omega_{max}$ 是系统中最高振动频率。

**推荐值**:

| 体系类型 | 温度 | 推荐珠子数 |
|----------|------|-----------|
| 含H体系 | 300K | 32-64 |
| 含H体系 | 100K | 64-128 |
| 含D体系 | 300K | 16-32 |
| 重原子 | 300K | 8-16 |

**收敛检查**:
```python
def check_bead_convergence(energies_vs_P):
    """检查珠子数收敛性"""
    for i in range(1, len(energies_vs_P)):
        diff = abs(energies_vs_P[i] - energies_vs_P[i-1])
        if diff < 0.001:  # 1 meV/atom
            return True, i
    return False, len(energies_vs_P)
```

### 时间步长

PIMD需要更小的时间步长：

- **经典MD**: 1-2 fs
- **PIMD**: 0.2-0.5 fs (因珠子间谐振弹簧需要)

### 恒温器选择

| 恒温器 | 适用场景 | 特点 |
|--------|----------|------|
| PILE-L | 平衡态计算 | 高效采样，推荐 |
| Nose-Hoover链 | 长时间模拟 | 稳定但较慢 |
| Langevin | 快速平衡 | 有随机噪声 |

---

## 后处理与分析

### 径向分布函数 (RDF)

```python
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# 读取轨迹
trajectory = read('water_pimd.pos_0.xyz', index=':')

# 计算RDF
from ase.geometry.analysis import Analysis

analysis = Analysis(trajectory)
rdf = analysis.get_rdf(rmax=6.0, nbins=100, 
                       elements=['O', 'H'])

# 绘制
r = np.linspace(0, 6, 100)
plt.plot(r, rdf)
plt.xlabel('r (Å)')
plt.ylabel('g(r)')
plt.title('O-H RDF from PIMD')
```

### 量子动能估计

PIMD提供两种动能估计：

1. **原始估计** (Primitive estimator):
   $$K_P = \frac{3NP}{2\beta} - \sum_{i=1}^P \frac{1}{2}m\omega_P^2(R_i - R_{i+1})^2$$

2. **维里估计** (Virial estimator, 推荐):
   $$K_V = \frac{3N}{2\beta} + \frac{1}{2P}\sum_{i=1}^P (R_i - R_c) \cdot \nabla V(R_i)$$

```python
def virial_kinetic_energy(positions, forces, temperature):
    """计算维里动能估计"""
    N = len(positions[0])  # 原子数
    P = len(positions)     # 珠子数
    
    # 质心位置
    R_c = np.mean(positions, axis=0)
    
    # 维里项
    virial = 0
    for i in range(P):
        virial += np.sum((positions[i] - R_c) * forces[i])
    virial /= P
    
    # 动能
    kB = 8.617333e-5  # eV/K
    kinetic = (3 * N / 2) * kB * temperature + 0.5 * virial
    
    return kinetic
```

### 与经典MD对比

```python
import matplotlib.pyplot as plt

# 读取PIMD和经典MD的RDF
r_pimd, rdf_pimd = load_rdf('pimd_rdf.dat')
r_classical, rdf_classical = load_rdf('classical_rdf.dat')

plt.plot(r_pimd, rdf_pimd, label='PIMD (Quantum)')
plt.plot(r_classical, rdf_classical, label='Classical MD')
plt.xlabel('r (Å)')
plt.ylabel('g(r)')
plt.legend()
plt.title('Quantum vs Classical RDF')
```

---

## 高级应用

### 环聚合物分子动力学 (RPMD)

RPMD是一种近似量子动力学方法：

```xml
<!-- i-PI RPMD输入 -->
<motion mode='dynamics'>
  <dynamics mode='nve'>
    <timestep units='femtosecond'>0.5</timestep>
  </dynamics>
</motion>
```

### 质心PIMD

用于计算量子热力学性质：

```xml
<ensemble>
  <temperature units='kelvin'>300</temperature>
</ensemble>
<motion mode='dynamics'>
  <dynamics mode='nvt'>
    <thermostat mode='pile_g'>
      <tau units='femtosecond'>100</tau>
    </thermostat>
  </dynamics>
</motion>
```

### 机器学习势加速PIMD

使用ML势替代昂贵的DFT计算：

```xml
<ffsocket name='ml_potential' mode='unix'>
  <address>ml_pot</address>
</ffsocket>

<system>
  <forces>
    <force forcefield='ml_potential'/>
  </forces>
</system>
```

---

## 注意事项

1. **计算成本**: PIMD计算量为经典MD的P倍，需要大量计算资源
2. **收敛性**: 珠子数必须充分收敛，特别是轻原子体系
3. **时间步长**: 珠子间谐振弹簧需要更小的时间步长
4. **初始平衡**: 需要充分的热化时间使环聚合物达到平衡
5. **并行效率**: 珠子间可并行，理想情况下加速比接近P

---

## 参考资源

- i-PI官网: https://ipi-code.org/
- PIMD (JAEA): https://ccse.jaea.go.jp/software/PIMD/
- ABINIT PIMD: https://docs.abinit.org/topics/PIMD/
- 文献: Marx & Hutter, "Ab Initio Molecular Dynamics", Cambridge (2009)
