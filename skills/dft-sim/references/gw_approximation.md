# GW近似方法详解

## 1. 原理简述

### 1.1 什么是GW近似

GW近似是一种超越DFT的准粒子能带计算方法，用于更准确地描述电子关联效应。在GW近似中：

- **G**: 格林函数 (Green's function)
- **W**: 屏蔽库仑相互作用 (Screened Coulomb interaction)

自能(Self-energy)表示为：
```
Σ = iGW
```

其中G是单粒子格林函数，W是动态屏蔽库仑相互作用。

### 1.2 GW近似的层次

| 方法 | 描述 | 计算成本 |
|------|------|---------|
| **G0W0** | 单发计算，使用DFT轨道和本征值计算G和W | 低 |
| **GW0** | 在G中自洽更新本征值，W保持DFT值 | 中 |
| **G0W0R** | G0W0 + RPA响应函数 | 中 |
| **scGW0** | 准粒子自洽GW0 | 高 |
| **scGW** | 完全自洽GW (本征值+轨道) | 很高 |
| **QPGW0** | 包含非对角自能项的GW0 | 高 |
| **QPGW** | 包含非对角自能项的完全自洽GW | 很高 |
| **EVGW0** | 仅本征值自洽的GW0 (VASP默认) | 中 |
| **EVGW** | 仅本征值自洽的GW | 高 |

### 1.3 计算流程

```
步骤1: DFT自洽计算 → 获得波函数和电荷密度
    ↓
步骤2: 准备GW计算 (可选: 计算响应函数)
    ↓
步骤3: GW计算 → 获得准粒子能级
    ↓
步骤4: 后处理分析 (能带图、带隙等)
```

### 1.4 关键参数

- **NOMEGA**: 频率点数，控制能量积分精度
- **ENCUTGW**: GW计算的平面波截断能
- **NBANDSGW**: GW计算的能带数
- **NELMGW**: GW自洽迭代次数

---

## 2. VASP实现

### 2.1 G0W0计算 (单发)

**INCAR设置**:
```
# 基础参数
SYSTEM = Si G0W0
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400

# GW参数
ALGO = G0W0       # 或 EVGW0 (VASP 6.x)
NBANDS = 64       # 总能带数，建议设为2-3倍价带数
NOMEGA = 50       # 频率点数，默认50

# 可选: 限制GW计算的能带范围
# NBANDSGW = 20   # 只计算前20个能带的GW修正
# NBANDSO = 4     # 价带数
# NBANDSV = 4     # 导带数
```

**计算步骤**:
```bash
# 步骤1: DFT自洽计算
# ISMEAR = -5 (四面体方法用于半导体/绝缘体)
# 或 ISMEAR = 0; SIGMA = 0.05 (高斯展宽用于金属)
mpirun -np 16 vasp_std

# 步骤2: GW计算
# 修改INCAR，设置ALGO = G0W0
# 确保WAVECAR存在
mpirun -np 16 vasp_std
```

### 2.2 GW0计算 (部分自洽)

**INCAR设置**:
```
# 基础参数
SYSTEM = Si GW0
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400

# GW0参数
ALGO = GW0        # 或 EVGW0
NBANDS = 64
NOMEGA = 50

# 自洽设置
NELMGW = 4        # GW自洽迭代次数 (VASP 6.3+)
# NELM = 4        # VASP 6.2及更早版本使用
```

### 2.3 scGW0计算 (准粒子自洽)

**INCAR设置**:
```
# 基础参数
SYSTEM = Si scGW0
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400

# scGW0参数
ALGO = QPGW0      # 包含非对角自能项
NBANDS = 64
NOMEGA = 50
NELMGW = 4

# 可选: 使用COHSEX近似加速收敛
# NOMEGA = 1      # 静态COHSEX近似
```

### 2.4 scGW计算 (完全自洽)

**INCAR设置**:
```
# 基础参数
SYSTEM = Si scGW
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400

# scGW参数
ALGO = QPGW       # 或 scGW (旧版本)
NBANDS = 64
NOMEGA = 50
NELMGW = 4

# 混合参数 (用于密度混合)
# IMIX = 0        # 使用damped MD算法
# TIME = 0.4      # 步长
```

### 2.5 关键参数详解

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `ALGO` | GW算法类型 | G0W0, GW0, EVGW0, QPGW0, QPGW |
| `NBANDS` | 总能带数 | 2-3倍价带数，或平面波数 |
| `NOMEGA` | 频率点数 | 50-100 |
| `NELMGW` | GW迭代次数 | 1(G0W0), 4-6(GW0/scGW) |
| `ENCUTGW` | GW截断能 | ENCUT或更低 |
| `LSPECTRAL` | 使用谱方法 | .TRUE. (推荐) |
| `OMEGAMAX` | 最大频率 | -1 (自动) |
| `OMEGAMIN` | 最小频率 | 自动确定 |
| `NOMEGAR` | 实频率点数 | 与NOMEGA相同 |
| `ENCUTGWSO` | 自旋轨道耦合截断 | 与ENCUTGW相同 |
| `NBANDSGW` | GW计算的能带数 | 通常等于NBANDS |
| `NBANDSO` | 价带数 | 体系依赖 |
| `NBANDSV` | 导带数 | 体系依赖 |

### 2.6 收敛性参数

```
# 频率收敛
NOMEGA = 50         # 起点
NOMEGA = 100        # 测试收敛

# 能带收敛
NBANDS = 64         # 起点
NBANDS = 128        # 测试收敛
NBANDS = 256        # 严格收敛

# 截断能收敛
ENCUTGW = 200       # 起点
ENCUTGW = 300       # 测试收敛
ENCUTGW = 400       # 严格收敛
```

---

## 3. Quantum ESPRESSO实现

QE本身不直接支持GW计算，需要通过以下插件:
- **Yambo**: 最常用的GW代码
- **Wannier90**: 用于能带插值
- **GWW**: QE自带的GW代码 (较少使用)

### 3.1 Yambo安装与配置

```bash
# 下载Yambo
git clone https://github.com/yambo-code/yambo.git
cd yambo

# 配置
./configure --with-qe=/path/to/qe \
            --enable-open-mp \
            --enable-mpi

# 编译
make all -j$(nproc)
```

### 3.2 Yambo G0W0计算流程

**步骤1: QE SCF计算** (`scf.in`):
```fortran
&CONTROL
  calculation = 'scf'
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
  occupations = 'fixed'
/
&ELECTRONS
  conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (alat)
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS automatic
  6 6 6 0 0 0
```

**步骤2: QE NSCF计算** (`nscf.in`):
```fortran
&CONTROL
  calculation = 'nscf'
  prefix = 'si'
  outdir = './tmp'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  nbnd = 100
  occupations = 'fixed'
/
&ELECTRONS
  conv_thr = 1.0d-10
/
ATOMIC_SPECIES
  Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (alat)
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS automatic
  6 6 6 0 0 0
```

**步骤3: Yambo初始化**:
```bash
# 生成Yambo数据库
yambo -F setup.in -J si_g0w0

# 编辑setup.in
vim setup.in
```

**setup.in**:
```
setup
# |RL| =  1000  # 实空间格点数 (自动设置)
```

**步骤4: Yambo GW输入** (`gw.in`):
```
gw0                          # [R GW] GoWo Quasiparticle energy levels
ppa                          # [R Xp] Plasmon Pole Approximation
HF_and_locXC                 # [R XX] Hartree-Fock Self-energy and local XC
em1d                         # [R Xd] Dynamical Inverse Dielectric Matrix

EXXRLvcs= 40         Ry      # [XX] Exchange    RL components
% BndsRnXp
  1 | 100 |                   # [Xp] Polarization function bands
%
NGsBlkXp= 4          Ry      # [Xp] Response block size
% GbndRnge
  1 | 100 |                   # [GW] GW bands range
%
GDamping= 0.10000    eV      # [GW] G
damping
DysSolver= "n"               # [GW] Dyson Equation solver (n,s,g)
% QPkrange                    # [GW] QP generalized Kpoint/Band indices
  1|  28|  1|  10|
%
```

**步骤5: 运行Yambo**:
```bash
# 运行GW计算
yambo -F gw.in -J si_g0w0

# 或使用MPI并行
mpirun -np 16 yambo -F gw.in -J si_g0w0
```

### 3.3 Yambo关键参数详解

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `EXXRLvcs` | 交换项截断 | 2-4倍ecutwfc |
| `BndsRnXp` | 极化函数能带范围 | 1-NBANDS |
| `NGsBlkXp` | 响应块大小 | 2-4 Ry |
| `GbndRnge` | GW能带范围 | 1-NBANDS |
| `GDamping` | G阻尼 | 0.1 eV |
| `ppa` | 使用等离子体极点近似 | 推荐用于大体系 |
| `QPkrange` | 计算的k点/能带范围 | 体系依赖 |

### 3.4 Yambo收敛性测试

**能带收敛**:
```
# 测试不同的BndsRnXp
% BndsRnXp
  1 |  50 |        # 测试1
  1 | 100 |        # 测试2
  1 | 200 |        # 测试3
%
```

**响应块收敛**:
```
# 测试不同的NGsBlkXp
NGsBlkXp= 2  Ry    # 测试1
NGsBlkXp= 4  Ry    # 测试2
NGsBlkXp= 6  Ry    # 测试3
```

**k点收敛**:
```
# 在nscf中测试不同的k点网格
K_POINTS automatic
  6 6 6 0 0 0      # 测试1
  8 8 8 0 0 0      # 测试2
  10 10 10 0 0 0   # 测试3
```

---

## 4. 完整输入文件示例

### 4.1 VASP: Si G0W0计算

**目录结构**:
```
Si_G0W0/
├── 01_dft/           # DFT自洽计算
├── 02_g0w0/          # G0W0计算
└── 03_analysis/      # 后处理分析
```

**01_dft/INCAR**:
```
SYSTEM = Si DFT
ISMEAR = -5
ENCUT = 400
EDIFF = 1E-8
LORBIT = 11

# 输出波函数
LWAVE = .TRUE.
LCHARG = .TRUE.
```

**01_dft/POSCAR**:
```
Si
   5.43
     0.5000000000000000    0.5000000000000000    0.0000000000000000
     0.0000000000000000    0.5000000000000000    0.5000000000000000
     0.5000000000000000    0.0000000000000000    0.5000000000000000
   2
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.2500000000000000  0.2500000000000000  0.2500000000000000
```

**01_dft/KPOINTS**:
```
Automatic mesh
0
Gamma
6 6 6
0 0 0
```

**02_g0w0/INCAR**:
```
SYSTEM = Si G0W0
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400

# GW参数
ALGO = G0W0
NBANDS = 64
NOMEGA = 50

# 读取DFT波函数
ISTART = 1
ICHARG = 0
```

### 4.2 Yambo: Si G0W0计算

**完整脚本** (`run_g0w0.sh`):
```bash
#!/bin/bash

# 步骤1: SCF计算
mpirun -np 8 pw.x -in scf.in > scf.out

# 步骤2: NSCF计算
mpirun -np 8 pw.x -in nscf.in > nscf.out

# 步骤3: Yambo初始化
yambo -F setup.in -J si_g0w0

# 步骤4: 生成GW输入
yambo -p p -g n -V all -F gw.in -J si_g0w0

# 步骤5: 运行GW
mpirun -np 16 yambo -F gw.in -J si_g0w0

# 步骤6: 提取结果
yambo -Q -J si_g0w0
```

---

## 5. 结果分析脚本

### 5.1 VASP结果提取

**提取准粒子能级** (`extract_qp.py`):
```python
#!/usr/bin/env python3
"""
提取VASP GW计算的准粒子能级
"""

import re
import numpy as np
import matplotlib.pyplot as plt

def extract_qp_energies(outcar_path='OUTCAR'):
    """从OUTCAR提取准粒子能级"""
    
    qp_data = []
    kpoints = []
    
    with open(outcar_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 查找k点信息
        if 'k-point' in line and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                k_coords = parts[1].split()
                if len(k_coords) >= 3:
                    kpoints.append([float(k_coords[0]), 
                                   float(k_coords[1]), 
                                   float(k_coords[2])])
        
        # 查找准粒子数据
        if 'band No.' in line and 'KS-energies' in line:
            # 读取表头后的数据
            i += 1
            while i < len(lines) and lines[i].strip():
                parts = lines[i].split()
                if len(parts) >= 6 and parts[0].isdigit():
                    band = int(parts[0])
                    ks_energy = float(parts[1])
                    qp_energy = float(parts[2])
                    sigma = float(parts[3])
                    vxc = float(parts[4])
                    Z = float(parts[6]) if len(parts) > 6 else 0.0
                    
                    qp_data.append({
                        'band': band,
                        'kpoint': len(kpoints),
                        'ks_energy': ks_energy,
                        'qp_energy': qp_energy,
                        'sigma': sigma,
                        'vxc': vxc,
                        'Z': Z
                    })
                i += 1
        
        i += 1
    
    return qp_data, kpoints

def calculate_bandgap(qp_data):
    """计算带隙"""
    
    # 按k点分组
    kpoints_dict = {}
    for d in qp_data:
        kp = d['kpoint']
        if kp not in kpoints_dict:
            kpoints_dict[kp] = []
        kpoints_dict[kp].append(d)
    
    min_gap = float('inf')
    max_vbm = -float('inf')
    min_cbm = float('inf')
    
    for kp, bands in kpoints_dict.items():
        # 排序
        bands_sorted = sorted(bands, key=lambda x: x['qp_energy'])
        
        # 找到HOMO和LUMO (简化处理)
        occupied = [b for b in bands_sorted if b['qp_energy'] < 0]
        unoccupied = [b for b in bands_sorted if b['qp_energy'] >= 0]
        
        if occupied and unoccupied:
            vbm = max(b['qp_energy'] for b in occupied)
            cbm = min(b['qp_energy'] for b in unoccupied)
            gap = cbm - vbm
            
            max_vbm = max(max_vbm, vbm)
            min_cbm = min(min_cbm, cbm)
            min_gap = min(min_gap, gap)
    
    return {
        'direct_gap': min_gap,
        'vbm': max_vbm,
        'cbm': min_cbm,
        'indirect_gap': min_cbm - max_vbm
    }

def plot_qp_vs_ks(qp_data, output='qp_vs_ks.png'):
    """绘制准粒子能量vs KS能量"""
    
    ks_energies = [d['ks_energy'] for d in qp_data]
    qp_energies = [d['qp_energy'] for d in qp_data]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(ks_energies, qp_energies, alpha=0.5)
    
    # 绘制y=x参考线
    min_e = min(min(ks_energies), min(qp_energies))
    max_e = max(max(ks_energies), max(qp_energies))
    plt.plot([min_e, max_e], [min_e, max_e], 'r--', label='y=x')
    
    plt.xlabel('KS Energy (eV)')
    plt.ylabel('QP Energy (eV)')
    plt.title('Quasiparticle vs KS Energies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Plot saved to {output}")

if __name__ == '__main__':
    import sys
    
    outcar = sys.argv[1] if len(sys.argv) > 1 else 'OUTCAR'
    
    print(f"Extracting QP energies from {outcar}...")
    qp_data, kpoints = extract_qp_energies(outcar)
    
    print(f"\nFound {len(qp_data)} QP states at {len(kpoints)} k-points")
    
    # 计算带隙
    gap_info = calculate_bandgap(qp_data)
    print(f"\nBand Gap Analysis:")
    print(f"  VBM: {gap_info['vbm']:.3f} eV")
    print(f"  CBM: {gap_info['cbm']:.3f} eV")
    print(f"  Direct gap: {gap_info['direct_gap']:.3f} eV")
    print(f"  Indirect gap: {gap_info['indirect_gap']:.3f} eV")
    
    # 保存数据
    with open('qp_energies.dat', 'w') as f:
        f.write("# band kpoint KS_energy QP_energy sigma Vxc Z\n")
        for d in qp_data:
            f.write(f"{d['band']} {d['kpoint']} {d['ks_energy']:.6f} "
                   f"{d['qp_energy']:.6f} {d['sigma']:.6f} "
                   f"{d['vxc']:.6f} {d['Z']:.6f}\n")
    
    print("\nData saved to qp_energies.dat")
    
    # 绘图
    plot_qp_vs_ks(qp_data)
```

### 5.2 Yambo结果提取

**提取Yambo结果** (`extract_yambo.py`):
```python
#!/usr/bin/env python3
"""
提取Yambo GW计算结果
"""

import re
import numpy as np
import matplotlib.pyplot as plt

def extract_qp_yambo(output_file='o-si_g0w0.qp'):
    """从Yambo输出提取准粒子能级"""
    
    qp_data = []
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        
        parts = line.split()
        if len(parts) >= 6:
            # Yambo格式: k-point band Eo Eqp Sc Vxc
            kpoint = int(parts[0])
            band = int(parts[1])
            eo = float(parts[2])      # KS能量
            eqp = float(parts[3])     # QP能量
            sc = float(parts[4])      # 自能
            vxc = float(parts[5])     # 交换关联势
            
            qp_data.append({
                'kpoint': kpoint,
                'band': band,
                'ks_energy': eo,
                'qp_energy': eqp,
                'sigma': sc,
                'vxc': vxc
            })
    
    return qp_data

def plot_band_structure(qp_data, output='bands_gw.png'):
    """绘制GW能带结构"""
    
    # 按能带分组
    bands_dict = {}
    for d in qp_data:
        b = d['band']
        if b not in bands_dict:
            bands_dict[b] = {'k': [], 'e': []}
        bands_dict[b]['k'].append(d['kpoint'])
        bands_dict[b]['e'].append(d['qp_energy'])
    
    plt.figure(figsize=(10, 6))
    
    for band, data in sorted(bands_dict.items()):
        k_indices = np.arange(len(data['k']))
        plt.plot(k_indices, data['e'], 'b-', linewidth=1)
    
    plt.xlabel('k-point index')
    plt.ylabel('Energy (eV)')
    plt.title('GW Band Structure')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Band structure saved to {output}")

if __name__ == '__main__':
    import sys
    
    qp_file = sys.argv[1] if len(sys.argv) > 1 else 'o-si_g0w0.qp'
    
    print(f"Extracting QP energies from {qp_file}...")
    qp_data = extract_qp_yambo(qp_file)
    
    print(f"Found {len(qp_data)} QP states")
    
    # 保存数据
    with open('qp_yambo.dat', 'w') as f:
        f.write("# kpoint band KS_energy QP_energy sigma Vxc\n")
        for d in qp_data:
            f.write(f"{d['kpoint']} {d['band']} {d['ks_energy']:.6f} "
                   f"{d['qp_energy']:.6f} {d['sigma']:.6f} {d['vxc']:.6f}\n")
    
    print("Data saved to qp_yambo.dat")
    
    # 绘图
    plot_band_structure(qp_data)
```

---

## 6. 常见错误和解决方案

### 6.1 VASP常见错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `ERROR: number of bands NBANDS` | 能带数设置不当 | 增加NBANDS到平面波数 |
| `ERROR: not enough bands` | 能带数不足 | NBANDS ≥ 2×价带数 |
| `WARNING: Sub-Space-Matrix is not hermitian` | 数值不稳定 | 增加ENCUT或调整ALGO |
| `ERROR: Fermi level cannot be determined` | 金属体系设置不当 | 使用ISMEAR=0; SIGMA=0.05 |
| `Out of memory` | 内存不足 | 减少NBANDS或增加并行数 |
| `ERROR: WAVECAR not found` | 缺少波函数文件 | 先运行DFT计算 |
| `Convergence not reached` | 自洽不收敛 | 增加NELMGW或调整mixing |

### 6.2 Yambo常见错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `Database not found` | 缺少QE数据库 | 检查outdir路径 |
| `Inconsistent k-grid` | k点网格不匹配 | 确保nscf使用完整BZ |
| `Memory allocation failed` | 内存不足 | 减少BndsRnXp或增加并行 |
| `Convergence not reached` | 迭代不收敛 | 增加GDamping |
| `FFT mesh too small` | FFT网格不足 | 增加ecutrho |

### 6.3 收敛性问题

**问题**: GW计算不收敛

**解决方案**:
1. 增加NOMEGA到100或更高
2. 使用LSPECTRAL=.TRUE. (VASP)
3. 调整GDamping参数 (Yambo)
4. 使用COHSEX近似作为初始猜测
5. 增加ENCUTGW

### 6.4 带隙低估/高估

**问题**: GW带隙与实验不符

**可能原因**:
1. k点网格不够密
2. 能带数不足
3. 截断能不够高
4. 需要更高阶的GW (scGW)

**解决方案**:
1. 增加k点密度
2. 测试NBANDS收敛
3. 测试ENCUTGW收敛
4. 考虑使用GW0或scGW

---

## 7. 参考文献

### 7.1 基础理论

1. **Hedin, L. (1965)**. New Method for Calculating the One-Particle Green's Function with Application to the Electron-Gas Problem. *Physical Review*, 139(3A), A796.

2. **Aryasetiawan, F., & Gunnarsson, O. (1998)**. The GW method. *Reports on Progress in Physics*, 61(3), 237.

3. **Onida, G., Reining, L., & Rubio, A. (2002)**. Electronic excitations: density-functional versus many-body Green's-function approaches. *Reviews of Modern Physics*, 74(2), 601.

### 7.2 VASP GW

4. **Shishkin, M., Kresse, G. (2006)**. Implementation and performance of the frequency-dependent GW method within the PAW framework. *Physical Review B*, 74(3), 035101.

5. **VASP Wiki**: https://www.vasp.at/wiki/index.php/GW_approximation

6. **VASP Practical Guide to GW**: https://www.vasp.at/wiki/index.php/Practical_guide_to_GW_calculations

### 7.3 Yambo

7. **Marini, A., Hogan, C., Grüning, M., & Varsano, D. (2009)**. yambo: An ab initio tool for excited state calculations. *Computer Physics Communications*, 180(8), 1392-1403.

8. **Sangalli, D., et al. (2019)**. Many-body perturbation theory calculations using the yambo code. *Journal of Physics: Condensed Matter*, 31(32), 325902.

9. **Yambo Documentation**: https://www.yambo-code.eu/

### 7.4 应用案例

10. **Rangel, T., et al. (2012)**. Band structure of gold from many-body perturbation theory. *Physical Review B*, 86(12), 125125.

11. **van Setten, M. J., et al. (2015)**. GW method for extended systems: A tutorial review. *Journal of Chemical Theory and Computation*, 11(7), 3115-3127.

12. **Reining, L. (2018)**. The GW approximation: content, successes and limitations. *Wiley Interdisciplinary Reviews: Computational Molecular Science*, 8(5), e1344.

---

## 8. 最佳实践

### 8.1 计算流程建议

1. **从DFT开始**: 确保DFT计算收敛且合理
2. **收敛性测试**: 系统测试NBANDS、NOMEGA、k点
3. **分步计算**: 先做G0W0，再考虑GW0/scGW
4. **验证结果**: 与实验值和文献对比

### 8.2 参数选择指南

| 体系类型 | NBANDS | NOMEGA | ENCUTGW | 推荐方法 |
|---------|--------|--------|---------|---------|
| 小分子 | 2-3×价带 | 50 | ENCUT | G0W0 |
| 半导体 | 3-4×价带 | 50-100 | ENCUT | G0W0/GW0 |
| 金属 | 平面波数 | 100 | ENCUT | G0W0 |
| 强关联 | 4-5×价带 | 100 | 1.5×ENCUT | scGW0 |
| 2D材料 | 3-4×价带 | 50-100 | ENCUT | G0W0 + 截断 |

### 8.3 计算资源估算

GW计算资源需求:
- **内存**: ~NBANDS² × NOMEGA × 8 bytes
- **CPU时间**: 比DFT高10-100倍
- **推荐**: 使用高性能计算集群

---

*文档版本: 1.0*
*更新日期: 2026-03-08*
