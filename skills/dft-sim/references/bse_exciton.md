# BSE激子计算方法详解

## 1. 原理简述

### 1.1 什么是BSE

Bethe-Salpeter方程(BSE)是描述电子-空穴对(激子)相互作用的多体微扰理论方法。它基于两粒子格林函数，可以准确描述:

- **激子束缚态**: 电子-空穴吸引形成的束缚态
- **光学吸收谱**: 包含激子效应的介电函数
- **激子波函数**: 电子-空穴相对运动的空间分布

### 1.2 BSE方程形式

在Tamm-Dancoff近似(TDA)下，BSE本征值方程为:

```
∑_{v'c'k'} H^{BSE}_{vck,v'c'k'} A^{λ}_{v'c'k'} = Ω^{λ} A^{λ}_{vck}
```

其中哈密顿量为:
```
H^{BSE}_{vck,v'c'k'} = (E_{ck} - E_{vk})δ_{vv'}δ_{cc'}δ_{kk'} 
                     - K^{dir}_{vck,v'c'k'} + K^{exc}_{vck,v'c'k'}
```

**各项含义**:
- **对角项**: 准粒子能量差 (E_c - E_v)
- **K^{dir}**: 直接项 (屏蔽库仑吸引) - 形成激子的关键
- **K^{exc}**: 交换项 (未屏蔽库仑排斥) - 单态-三重态分裂

### 1.3 BSE计算层次

| 方法 | 描述 | 适用场景 |
|------|------|---------|
| **IP** | 独立粒子近似 (RPA) | 无激子效应的参考 |
| **RPA** | 随机相近似 | 弱激子效应体系 |
| **BSE@DFT** | BSE使用DFT轨道和能量 | 快速估算 |
| **BSE@GW** | BSE使用GW准粒子能量 | 标准激子计算 |
| **BSE+TDA** | Tamm-Dancoff近似 | 大多数情况 |
| **Full BSE** | 包含耦合项 | 需要高精度时 |

### 1.4 计算流程

```
步骤1: DFT计算 → 获得基态波函数
    ↓
步骤2: GW计算 (可选但推荐) → 准粒子能量
    ↓
步骤3: 计算屏蔽库仑相互作用 W
    ↓
步骤4: 构建BSE哈密顿量并求解
    ↓
步骤5: 计算介电函数和光学谱
```

---

## 2. VASP实现

### 2.1 基础BSE计算

**前置要求**: 必须先完成GW计算获得Wxxxx.tmp文件

**INCAR设置**:
```
# 基础参数
SYSTEM = Si BSE
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400
NBANDS = 64       # 与GW计算相同

# BSE参数
ALGO = BSE        # BSE算法
NBANDSO = 4       # 价带数 (包含在BSE中)
NBANDSV = 4       # 导带数 (包含在BSE中)
OMEGAMAX = 10     # 最大激发能量 (eV)

# 可选: 限制计算的能带范围
# NBANDSGW = 20   # GW计算的能带数
```

**计算步骤**:
```bash
# 步骤1: DFT自洽计算
mpirun -np 16 vasp_std

# 步骤2: GW计算 (获得Wxxxx.tmp)
# ALGO = G0W0; NBANDS = 64
mpirun -np 16 vasp_std

# 步骤3: BSE计算
# 修改INCAR: ALGO = BSE
mpirun -np 16 vasp_std
```

### 2.2 TDHF计算 (无需GW)

**INCAR设置**:
```
# TDHF不需要GW的W文件
SYSTEM = Si TDHF
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400

# TDHF参数
ALGO = TDHF       # 时间依赖Hartree-Fock
NBANDSO = 4
NBANDSV = 4
OMEGAMAX = 10

# 杂化泛函轨道 (可选)
# LHFCALC = .TRUE.
# HFSCREEN = 0.2
```

### 2.3 关键参数详解

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `ALGO` | 算法类型 | BSE, TDHF |
| `NBANDSO` | BSE价带数 | 2-4个最高价带 |
| `NBANDSV` | BSE导带数 | 2-4个最低导带 |
| `OMEGAMAX` | 最大激发能 | 覆盖目标能量范围 |
| `ANTIRES` | 反共振项 | 0 (TDA), 2 (full BSE) |
| `LADDER` | 包含交换项 | .TRUE. (默认) |
| `LTRIPLET` | 计算三重态 | .FALSE. (默认) |

### 2.4 收敛性参数

```
# 能带收敛测试
NBANDSO = 2    # 测试1
NBANDSO = 4    # 测试2
NBANDSO = 6    # 测试3

NBANDSV = 2    # 测试1
NBANDSV = 4    # 测试2
NBANDSV = 6    # 测试3

# k点收敛 (在DFT步骤中)
# KPOINTS
# 4 4 4        # 测试1
# 6 6 6        # 测试2
# 8 8 8        # 测试3
```

### 2.5 输出文件解析

**vasprun.xml** 包含:
- 激子能量 (excitation energies)
- 振子强度 (oscillator strengths)
- 介电函数 (dielectric function)

**OUTCAR** 包含:
- BSE哈密顿量信息
- 收敛信息

---

## 3. Quantum ESPRESSO + Yambo实现

### 3.1 计算流程

```bash
# 步骤1: DFT SCF
mpirun -np 8 pw.x -in scf.in > scf.out

# 步骤2: DFT NSCF (更多能带和k点)
mpirun -np 8 pw.x -in nscf.in > nscf.out

# 步骤3: Yambo初始化
yambo -F setup.in -J bse_calc

# 步骤4: 计算屏蔽库仑相互作用 (GW步骤)
yambo -p p -g n -F gw.in -J bse_calc

# 步骤5: BSE计算
yambo -y d -o b -k sex -F bse.in -J bse_calc

# 步骤6: 后处理分析
ypp -F exciton.in -J bse_calc
```

### 3.2 QE输入文件

**scf.in**:
```fortran
&CONTROL
  calculation = 'scf'
  prefix = 'hBN'
  outdir = './tmp'
  pseudo_dir = './pseudo'
/
&SYSTEM
  ibrav = 4
  celldm(1) = 4.733
  celldm(3) = 2.5
  nat = 4
  ntyp = 2
  ecutwfc = 80
  ecutrho = 640
  occupations = 'fixed'
/
&ELECTRONS
  conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  B  10.811  B.pbe-n-kjpaw_psl.1.0.0.UPF
  N  14.007  N.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  B  0.000000000  0.000000000  0.000000000
  N  0.000000000  0.000000000  0.500000000
  B  0.333333333  0.666666667  0.000000000
  N  0.333333333  0.666666667  0.500000000
K_POINTS automatic
  12 12 6 0 0 0
```

**nscf.in**:
```fortran
&CONTROL
  calculation = 'nscf'
  prefix = 'hBN'
  outdir = './tmp'
/
&SYSTEM
  ibrav = 4
  celldm(1) = 4.733
  celldm(3) = 2.5
  nat = 4
  ntyp = 2
  ecutwfc = 80
  ecutrho = 640
  nbnd = 100
  occupations = 'fixed'
/
&ELECTRONS
  conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  B  10.811  B.pbe-n-kjpaw_psl.1.0.0.UPF
  N  14.007  N.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (crystal)
  B  0.000000000  0.000000000  0.000000000
  N  0.000000000  0.000000000  0.500000000
  B  0.333333333  0.666666667  0.000000000
  N  0.333333333  0.666666667  0.500000000
K_POINTS automatic
  18 18 6 0 0 0  ! 更密的k点用于BSE
```

### 3.3 Yambo BSE输入

**bse.in**:
```
optics                       # [R OPT] Optics
bse                          # [R BSE] Bethe Salpeter
bss                          # [R BSS] Bethe Salpeter Solver

% BSEBands
  4 |  7 |                   # [BSE] Bands range
%
% BSEkpts
  1 |  54 |                  # [BSE] K-points range (all)
%
BSENGexx=  40          Ry    # [BSE] Exchange components
BSENGBlk=  4           Ry    # [BSE] Screened interaction block size
% KfnQP_E
  2.870000 | 1.000000 | 1.000000 |  # [EXTQP BSK BSS] E parameters (scissor)
%
BSEprop= "abs"               # [BSS] Can be "abs", "kerr", "magn", "dich"
% BEnRange
  0.00000 | 10.00000 | eV    # [BSS] Energy range
%
% BDmRange
  0.10000 |  0.10000 | eV    # [BSS] Damping range
%
BEnSteps=  200               # [BSS] Energy steps
% BLongDir
  1.000000 | 0.000000 | 0.000000 |  # [BSS] [cc] Electric Field
%
BSSmod= "h"                  # [BSS] Solving method (h=Haydock, d=diagonalization, i=inversion)
```

### 3.4 Yambo关键参数详解

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `BSEBands` | BSE能带范围 | 价带顶±2-4, 导带底±2-4 |
| `BSENGexx` | 交换项截断 | 2-4倍ecutwfc |
| `BSENGBlk` | 屏蔽相互作用块大小 | 2-4 Ry |
| `KfnQP_E` | 准粒子能量修正 (剪刀算符) | GW带隙-DFT带隙 |
| `BSSmod` | 求解方法 | h=Haydock, d=对角化 |
| `BEnRange` | 能量范围 | 覆盖目标光谱范围 |
| `BDmRange` | 展宽 | 0.05-0.2 eV |

### 3.5 求解方法选择

**Haydock方法** (`BSSmod= "h"`):
- 优点: 内存效率高，适合大体系
- 缺点: 不给出激子波函数
- 适用: 只需要光谱的情况

**对角化方法** (`BSSmod= "d"`):
- 优点: 获得激子能量和波函数
- 缺点: 内存需求大 O(N²)
- 适用: 需要分析激子性质

**反演方法** (`BSSmod= "i"`):
- 优点: 直接计算响应函数
- 缺点: 计算成本高
- 适用: 特殊应用

---

## 4. 完整输入文件示例

### 4.1 VASP: Si BSE计算

**目录结构**:
```
Si_BSE/
├── 01_dft/           # DFT自洽
├── 02_gw/            # GW计算
├── 03_bse/           # BSE计算
└── 04_analysis/      # 后处理
```

**01_dft/INCAR**:
```
SYSTEM = Si DFT
ISMEAR = -5
ENCUT = 400
EDIFF = 1E-8
LWAVE = .TRUE.
LCHARG = .TRUE.
```

**02_gw/INCAR**:
```
SYSTEM = Si GW
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400
ALGO = G0W0
NBANDS = 64
NOMEGA = 50
ISTART = 1
```

**03_bse/INCAR**:
```
SYSTEM = Si BSE
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400
NBANDS = 64

# BSE参数
ALGO = BSE
NBANDSO = 4
NBANDSV = 4
OMEGAMAX = 10

# 读取GW结果
ISTART = 1
```

### 4.2 Yambo: hBN BSE计算

**完整脚本** (`run_bse.sh`):
```bash
#!/bin/bash

# 设置
PREFIX="hBN"
NCPU=16

# 步骤1: SCF计算
echo "Running SCF..."
mpirun -np $NCPU pw.x -in scf.in > scf.out

# 步骤2: NSCF计算
echo "Running NSCF..."
mpirun -np $NCPU pw.x -in nscf.in > nscf.out

# 步骤3: Yambo初始化
echo "Initializing Yambo..."
yambo -F setup.in -J $PREFIX

# 步骤4: 计算屏蔽库仑相互作用
echo "Calculating screened Coulomb interaction..."
cat > gw.in << EOF
ppa
em1d
% BndsRnXp
  1 | 100 |
%
NGsBlkXp= 4 Ry
% GbndRnge
  1 | 100 |
%
EOF
yambo -F gw.in -J $PREFIX

# 步骤5: BSE计算
echo "Running BSE..."
cat > bse.in << EOF
optics
bse
bss
% BSEBands
  4 | 7 |
%
BSENGexx= 40 Ry
BSENGBlk= 4 Ry
% KfnQP_E
  2.87 | 1.0 | 1.0 |
%
BSEprop= "abs"
% BEnRange
  0.0 | 10.0 | eV
%
BDmRange= 0.1 | 0.1 | eV
BEnSteps= 200
BSSmod= "d"
EOF
mpirun -np $NCPU yambo -F bse.in -J $PREFIX

# 步骤6: 激子分析
echo "Analyzing excitons..."
cat > exciton.in << EOF
excitons
% States
  1 | 10 |
%
% Degen_Step
  0.0100 | eV
%
EOF
ypp -F exciton.in -J $PREFIX

echo "BSE calculation completed!"
```

---

## 5. 结果分析脚本

### 5.1 VASP结果提取

**提取激子信息** (`extract_bse_vasp.py`):
```python
#!/usr/bin/env python3
"""
提取VASP BSE计算的激子信息
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def extract_bse_vasprun(vasprun_path='vasprun.xml'):
    """从vasprun.xml提取BSE结果"""
    
    tree = ET.parse(vasprun_path)
    root = tree.getroot()
    
    # 提取激子能量和振子强度
    excitons = []
    
    # 查找变分步骤
    for varray in root.findall('.//varray'):
        name = varray.get('name')
        if name == 'exciton_energies':
            for v in varray.findall('v'):
                energies = [float(x) for x in v.text.split()]
                excitons.extend(energies)
    
    # 提取介电函数
    dielectric = []
    for dielectric_tag in root.findall('.//dielectricfunction'):
        for array in dielectric_tag.findall('array'):
            for set_tag in array.findall('set'):
                for r in set_tag.findall('r'):
                    values = [float(x) for x in r.text.split()]
                    dielectric.append(values)
    
    return np.array(excitons), np.array(dielectric)

def plot_absorption_spectrum(dielectric, output='absorption.png'):
    """绘制吸收谱"""
    
    if len(dielectric) == 0:
        print("No dielectric data found")
        return
    
    energy = dielectric[:, 0]
    epsilon_imag = dielectric[:, 2]  # 虚部
    
    plt.figure(figsize=(10, 6))
    plt.plot(energy, epsilon_imag, 'b-', linewidth=2)
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Im[ε(ω)]', fontsize=12)
    plt.title('BSE Absorption Spectrum', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Absorption spectrum saved to {output}")

def analyze_excitons(excitons, gap=None):
    """分析激子信息"""
    
    print("\n" + "="*50)
    print("Exciton Analysis")
    print("="*50)
    
    # 排序激子能量
    sorted_exc = np.sort(excitons)
    
    print(f"\nNumber of excitons: {len(excitons)}")
    print(f"Lowest exciton energy: {sorted_exc[0]:.3f} eV")
    
    if gap:
        binding_energy = gap - sorted_exc[0]
        print(f"Band gap: {gap:.3f} eV")
        print(f"Exciton binding energy: {binding_energy:.3f} eV")
    
    print("\nFirst 10 exciton energies:")
    for i, e in enumerate(sorted_exc[:10]):
        print(f"  Exciton {i+1}: {e:.3f} eV")

if __name__ == '__main__':
    import sys
    
    vasprun = sys.argv[1] if len(sys.argv) > 1 else 'vasprun.xml'
    gap = float(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print(f"Extracting BSE data from {vasprun}...")
    excitons, dielectric = extract_bse_vasprun(vasprun)
    
    # 分析激子
    analyze_excitons(excitons, gap)
    
    # 绘制吸收谱
    if len(dielectric) > 0:
        plot_absorption_spectrum(dielectric)
    
    # 保存数据
    if len(excitons) > 0:
        np.savetxt('exciton_energies.dat', excitons, header='Exciton energy (eV)')
        print("\nExciton energies saved to exciton_energies.dat")
```

### 5.2 Yambo结果提取

**提取Yambo激子信息** (`extract_bse_yambo.py`):
```python
#!/usr/bin/env python3
"""
提取Yambo BSE计算的激子信息
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

def read_eps_file(filename='o-bse_calc.eps_q1_diago_bse'):
    """读取介电函数文件"""
    
    data = np.loadtxt(filename)
    energy = data[:, 0]
    eps_imag = data[:, 2]  # 虚部
    eps_real = data[:, 1]  # 实部
    
    return energy, eps_real, eps_imag

def read_exciton_file(filename='o-bse_calc.exc_q1_diago_bse'):
    """读取激子能量文件"""
    
    excitons = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                n = int(parts[0])
                energy = float(parts[1])
                strength = float(parts[2])
                excitons.append({'n': n, 'E': energy, 'f': strength})
    
    return excitons

def plot_bse_spectrum(energy, eps_imag, excitons=None, output='bse_spectrum.png'):
    """绘制BSE光谱"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制吸收谱
    ax.plot(energy, eps_imag, 'b-', linewidth=2, label='BSE')
    ax.fill_between(energy, eps_imag, alpha=0.3)
    
    # 标记激子峰
    if excitons:
        for exc in excitons[:5]:  # 前5个激子
            if exc['f'] > 0.01:  # 只标记强激子
                ax.axvline(x=exc['E'], color='r', linestyle='--', alpha=0.5)
                ax.annotate(f"E{exc['n']}={exc['E']:.2f}eV", 
                           xy=(exc['E'], max(eps_imag)*0.9),
                           fontsize=9, rotation=90, va='top')
    
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('Im[ε(ω)]', fontsize=12)
    ax.set_title('BSE Optical Absorption Spectrum', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Spectrum saved to {output}")

def analyze_excitons_yambo(excitons, gap=None):
    """分析Yambo激子结果"""
    
    print("\n" + "="*50)
    print("Yambo Exciton Analysis")
    print("="*50)
    
    print(f"\nNumber of excitons: {len(excitons)}")
    
    # 按能量排序
    sorted_exc = sorted(excitons, key=lambda x: x['E'])
    
    print(f"\nLowest exciton energy: {sorted_exc[0]['E']:.3f} eV")
    print(f"Oscillator strength: {sorted_exc[0]['f']:.4f}")
    
    if gap:
        binding_energy = gap - sorted_exc[0]['E']
        print(f"Band gap: {gap:.3f} eV")
        print(f"Exciton binding energy: {binding_energy:.3f} eV")
    
    print("\nFirst 10 excitons:")
    print(f"{'Index':<8} {'Energy (eV)':<15} {'Strength':<12}")
    print("-" * 35)
    for exc in sorted_exc[:10]:
        print(f"{exc['n']:<8} {exc['E']:<15.3f} {exc['f']:<12.4f}")

if __name__ == '__main__':
    import sys
    
    eps_file = sys.argv[1] if len(sys.argv) > 1 else 'o-bse_calc.eps_q1_diago_bse'
    exc_file = sys.argv[2] if len(sys.argv) > 2 else 'o-bse_calc.exc_q1_diago_bse'
    gap = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # 读取数据
    print(f"Reading dielectric function from {eps_file}...")
    energy, eps_real, eps_imag = read_eps_file(eps_file)
    
    print(f"Reading exciton data from {exc_file}...")
    excitons = read_exciton_file(exc_file)
    
    # 分析
    analyze_excitons_yambo(excitons, gap)
    
    # 绘图
    plot_bse_spectrum(energy, eps_imag, excitons)
    
    # 保存数据
    np.savetxt('bse_spectrum.dat', np.column_stack([energy, eps_real, eps_imag]),
               header='Energy(eV) Re[eps] Im[eps]')
    print("\nSpectrum data saved to bse_spectrum.dat")
```

### 5.3 激子波函数可视化

**Yambo激子波函数** (`plot_exciton_wf.py`):
```python
#!/usr/bin/env python3
"""
可视化Yambo激子波函数
需要ypp生成的激子波函数文件
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_exciton_weights(filename='o-bse_calc.exc_weights_at_q1'):
    """读取激子权重文件"""
    
    weights = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                kpt = int(parts[0])
                v_band = int(parts[1])
                c_band = int(parts[2])
                weight = float(parts[3])
                weights.append({
                    'kpt': kpt,
                    'v': v_band,
                    'c': c_band,
                    'w': weight
                })
    
    return weights

def plot_exciton_weights(weights, exciton_n=1, output='exciton_weights.png'):
    """绘制激子在k空间的权重分布"""
    
    # 筛选特定激子的贡献
    kpts = [w['kpt'] for w in weights]
    wvals = [w['w'] for w in weights]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(kpts, range(len(kpts)), c=wvals, cmap='hot', s=50)
    plt.colorbar(scatter, label='Weight')
    
    ax.set_xlabel('k-point index', fontsize=12)
    ax.set_ylabel('Transition index', fontsize=12)
    ax.set_title(f'Exciton {exciton_n} Weights in k-space', fontsize=14)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Exciton weights plot saved to {output}")

if __name__ == '__main__':
    import sys
    
    weights_file = sys.argv[1] if len(sys.argv) > 1 else 'o-bse_calc.exc_weights_at_q1'
    exciton_n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"Reading exciton weights from {weights_file}...")
    weights = read_exciton_weights(weights_file)
    
    print(f"Found {len(weights)} transitions")
    
    plot_exciton_weights(weights, exciton_n)
```

---

## 6. 常见错误和解决方案

### 6.1 VASP常见错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `WFULLxxxx.tmp not found` | 缺少GW的W文件 | 先运行GW计算 |
| `ERROR: NBANDSO + NBANDSV > NBANDS` | 能带设置错误 | 减少NBANDSO/V或增加NBANDS |
| `Out of memory in BSE` | BSE矩阵太大 | 减少NBANDSO/V |
| `ERROR: OMEGAMAX too large` | 激发能范围太大 | 减小OMEGAMAX |
| `No excitonic states found` | 参数设置不当 | 检查能带范围和能量窗口 |

### 6.2 Yambo常见错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `Database not found` | 缺少QE数据库 | 检查outdir路径 |
| `BSE matrix too large` | 内存不足 | 减少BSEBands范围 |
| `Screening not found` | 缺少GW步骤 | 先运行ppa/em1d计算 |
| `k-point grid incompatible` | k点不匹配 | 确保NSCF使用完整BZ |
| `Convergence not reached` | 迭代不收敛 | 调整BSSmod或增加迭代次数 |

### 6.3 收敛性问题

**问题**: 激子能量不收敛

**解决方案**:
1. 增加k点密度 (特别是NSCF步骤)
2. 增加BSE能带范围
3. 检查GW准粒子能量收敛
4. 调整BSENGBlk参数

### 6.4 激子束缚能异常

**问题**: 激子束缚能为负值

**可能原因**:
1. 剪刀算符设置错误
2. 能带顺序错误
3. 准粒子能量不准确

**解决方案**:
1. 检查KfnQP_E参数
2. 确认BSEBands包含正确的价带和导带
3. 验证GW计算结果

---

## 7. 参考文献

### 7.1 基础理论

1. **Rohlfing, M., & Louie, S. G. (2000)**. Electron-hole excitations and optical spectra from first principles. *Physical Review B*, 62(8), 4927.

2. **Onida, G., Reining, L., & Rubio, A. (2002)**. Electronic excitations: density-functional versus many-body Green's-function approaches. *Reviews of Modern Physics*, 74(2), 601.

3. **Benedict, L. X., Shirley, E. L., & Bohn, R. B. (1998)**. Optical absorption of insulators and the electron-hole interaction: An ab initio calculation. *Physical Review Letters*, 80(20), 4514.

### 7.2 VASP BSE

4. **VASP Wiki**: https://www.vasp.at/wiki/index.php/Bethe-Salpeter-equations_calculations

5. **Furthmüller, J., et al. (2006)**. *VASP Manual*, University of Vienna.

### 7.3 Yambo BSE

6. **Marini, A., Hogan, C., Grüning, M., & Varsano, D. (2009)**. yambo: An ab initio tool for excited state calculations. *Computer Physics Communications*, 180(8), 1392-1403.

7. **Sangalli, D., et al. (2019)**. Many-body perturbation theory calculations using the yambo code. *Journal of Physics: Condensed Matter*, 31(32), 325902.

8. **Yambo BSE Tutorial**: https://wiki.yambo-code.eu/wiki/index.php/BSE_tutorial_on_hBN

### 7.4 应用案例

9. **Wirtz, L., Marini, A., & Rubio, A. (2006)**. Excitons in boron nitride nanotubes: Dimensionality effects. *Physical Review Letters*, 96(12), 126104.

10. **Qiu, D. Y., da Jornada, F. H., & Louie, S. G. (2013)**. Optical spectrum of MoS2: many-body effects and diversity of exciton states. *Physical Review Letters*, 111(21), 216805.

11. **Grüning, M., Attaccalite, C., & Marini, A. (2010)**. Ab initio angle-resolved photoelectron spectroscopy of the 2H-MoS2. *Physica Status Solidi (b)*, 247(8), 2035-2039.

---

## 8. 最佳实践

### 8.1 计算流程建议

1. **从DFT开始**: 确保基态计算收敛
2. **GW修正**: 获得准确的准粒子能量
3. **收敛性测试**: 系统测试k点、能带、截断能
4. **激子分析**: 检查激子波函数和权重

### 8.2 参数选择指南

| 体系类型 | 价带数 | 导带数 | k点网格 | 推荐方法 |
|---------|--------|--------|---------|---------|
| 小分子 | 2-4 | 2-4 | 密集 | BSE@GW |
| 半导体 | 2-4 | 2-4 | 6×6×6+ | BSE@GW |
| 2D材料 | 2-4 | 2-4 | 18×18×1+ | BSE@GW |
| 表面 | 2-4 | 2-4 | 密集 | BSE@GW |

### 8.3 计算资源估算

BSE计算资源需求:
- **内存**: ~N_transitions² × 8 bytes
- **CPU时间**: 比GW高10-100倍
- **存储**: 激子波函数可能很大

### 8.4 典型激子束缚能参考

| 材料 | 激子束缚能 (eV) | 带隙 (eV) |
|------|----------------|----------|
| Si | 0.015 | 1.17 |
| GaAs | 0.004 | 1.52 |
| hBN | 0.7-2.0 | 5.9 |
| MoS2 | 0.5-0.9 | 1.8 |
| LiF | 1.5-2.0 | 14.2 |

---

*文档版本: 1.0*
*更新日期: 2026-03-08*
