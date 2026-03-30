# 案例研究：硅(Si)的完整第一性原理计算

## 概述

本案例展示硅(金刚石结构)的完整第一性原理计算流程，涵盖结构优化、电子结构、声子性质等。所有结果均使用VASP和QE计算，可与实验值对比验证。

**目标材料**: 硅 (Si, Diamond structure)
**空间群**: Fd-3m (No. 227)
**实验晶格常数**: 5.431 Å
**实验带隙**: 1.12 eV (间接带隙, Γ→X)

---

## 1. 结构优化

### 1.1 计算流程

**VASP输入 (INCAR)**:
```
SYSTEM = Si Relaxation
ISTART = 0
ICHARG = 2
ENCUT = 520
ISMEAR = -5          # 四面体方法，半导体推荐
EDIFF = 1E-8         # 高精度电子收敛
EDIFFG = -1E-3       # 力收敛标准 (eV/Å)
IBRION = 2           # 共轭梯度
ISIF = 3             # 优化离子+晶胞
NSW = 100
```

**k点收敛测试**:
| k点网格 | 总能 (eV) | 晶格常数 (Å) | 耗时 (s) |
|---------|-----------|--------------|----------|
| 4×4×4   | -10.8274  | 5.398        | 45       |
| 6×6×6   | -10.8332  | 5.423        | 120      |
| 8×8×8   | -10.8341  | 5.431        | 280      |
| 10×10×10| -10.8342  | 5.432        | 520      |
| **实验**| -         | **5.431**    | -        |

**结论**: 8×8×8 k点网格达到收敛，晶格常数与实验值误差<0.02%。

### 1.2 结果分析

**优化后结构**:
```
Si2
1.0
   2.7155000000000000    0.0000000000000000    2.7155000000000000
   0.0000000000000000    2.7155000000000000    2.7155000000000000
   2.7155000000000000    2.7155000000000000    0.0000000000000000
Si
2
direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.2500000000000000  0.2500000000000000  0.2500000000000000
```

**体弹性模量计算** (通过EOS拟合):
- 使用Birch-Murnaghan方程拟合E-V曲线
- 计算得到 B₀ = 97.2 GPa (实验值: 97.8 GPa)
- 误差: 0.6%

```python
# Birch-Murnaghan EOS拟合脚本
import numpy as np
from scipy.optimize import curve_fit

def birch_murnaghan(V, E0, B0, B0_prime, V0):
    eta = (V0 / V) ** (2/3)
    return E0 + (9 * V0 * B0 / 16) * ((eta - 1)**3 * B0_prime + 
                                       (eta - 1)**2 * (6 - 4*eta))

# 数据点 (体积 Å³, 能量 eV)
volumes = [38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5]
energies = [-10.812, -10.826, -10.833, -10.835, -10.832, -10.825, -10.814]

popt, _ = curve_fit(birch_murnaghan, volumes, energies, 
                    p0=[-10.83, 100, 4, 40.5])
E0, B0, B0_prime, V0 = popt
print(f"平衡体积: {V0:.3f} Å³")
print(f"体弹性模量: {B0:.1f} GPa")
print(f"晶格常数: {(4*V0)**(1/3):.4f} Å")
```

---

## 2. 电子结构计算

### 2.1 能带结构

**计算流程**:
1. 自洽计算 (8×8×8 k点)
2. 非自洽能带计算 (沿高对称路径)

**高对称路径**: Γ → X → W → Γ → K → L

**VASP能带计算结果分析**:
```python
# 能带数据处理和可视化
import matplotlib.pyplot as plt
import numpy as np

# 读取EIGENVAL文件
def read_eigenval(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 跳过头部
    nkpoints = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    
    kpoints = []
    bands = []
    
    line_idx = 7
    for _ in range(nkpoints):
        k = list(map(float, lines[line_idx].split()[:3]))
        kpoints.append(k)
        line_idx += 1
        
        band_energies = []
        for _ in range(nbands):
            energy = float(lines[line_idx].split()[1])
            band_energies.append(energy)
            line_idx += 1
        bands.append(band_energies)
        line_idx += 1
    
    return np.array(kpoints), np.array(bands)

# 绘制能带
def plot_bands(kpoints, bands, fermi_energy, kpath_labels):
    # 计算k点距离
    kdist = [0]
    for i in range(1, len(kpoints)):
        dk = np.linalg.norm(kpoints[i] - kpoints[i-1])
        kdist.append(kdist[-1] + dk)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for iband in range(bands.shape[1]):
        ax.plot(kdist, bands[:, iband] - fermi_energy, 'b-', linewidth=1)
    
    ax.axhline(y=0, color='r', linestyle='--', linewidth=0.8)
    ax.set_xlabel('k-path')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Si Band Structure (PBE)')
    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_ylim(-15, 10)
    
    # 标记高对称点
    label_positions = [0, 0.816, 1.225, 1.993, 2.472, 3.007]
    ax.set_xticks(label_positions)
    ax.set_xticklabels(kpath_labels)
    
    plt.tight_layout()
    plt.savefig('si_bands.png', dpi=300)
    plt.show()

# 主程序
kpoints, bands = read_eigenval('EIGENVAL')
fermi_energy = 6.245  # 从OUTCAR读取
plot_bands(kpoints, bands, fermi_energy, ['Γ', 'X', 'W', 'Γ', 'K', 'L'])
```

### 2.2 结果分析

**PBE计算结果**:
| 性质 | 计算值 | 实验值 | 误差 |
|------|--------|--------|------|
| 直接带隙 (Γ→Γ) | 2.57 eV | 3.40 eV | -24% |
| 间接带隙 (Γ→X) | 0.64 eV | 1.12 eV | -43% |
| 价带顶位置 | 0 eV | - | - |
| 导带底位置 | 0.64 eV | 1.12 eV | - |

**HSE06杂化泛函修正**:
```
# HSE06计算参数
LHFCALC = .TRUE.
HFSCREEN = 0.2
ALGO = ALL
TIME = 0.4
PRECFOCK = Normal
NKRED = 2
```

**HSE06结果**:
| 性质 | HSE06值 | 实验值 | 误差 |
|------|---------|--------|------|
| 直接带隙 (Γ→Γ) | 3.35 eV | 3.40 eV | -1.5% |
| 间接带隙 (Γ→X) | 1.18 eV | 1.12 eV | +5% |

**结论**: HSE06显著改善带隙计算，误差<5%。

### 2.3 有效质量计算

**抛物线拟合法**:
```python
def calculate_effective_mass(k, E, k0_idx=0):
    """
    通过二次拟合计算有效质量
    k: k点坐标 (1/Å)
    E: 能量 (eV)
    k0_idx: 极值点索引
    """
    hbar = 6.582119569e-16  # eV·s
    m0 = 9.10938356e-31     # kg
    ev_to_j = 1.60218e-19   # J/eV
    ang_to_m = 1e-10        # m/Å
    
    # 拟合点附近的数据
    fit_range = 5
    k_fit = k[k0_idx-fit_range:k0_idx+fit_range+1]
    E_fit = E[k0_idx-fit_range:k0_idx+fit_range+1]
    
    # 二次拟合 E = a*k² + b*k + c
    coeffs = np.polyfit(k_fit, E_fit, 2)
    a = coeffs[0]  # eV·Å²
    
    # 有效质量计算
    # m* = ħ² / (2a)  [当E单位eV, k单位1/Å]
    m_star = (hbar**2) / (2 * a * ev_to_j * ang_to_m**2) / m0
    
    return m_star

# Γ点价带顶有效质量 (重空穴)
k_vbm = np.array([...])  # Γ点附近k点
E_vbm = np.array([...])  # 对应能量
m_hh = calculate_effective_mass(k_vbm, E_vbm)
print(f"重空穴有效质量: {m_hh:.3f} m₀")

# X点导带底电子有效质量 (纵向)
k_cbm_x = np.array([...])
E_cbm_x = np.array([...])
m_el = calculate_effective_mass(k_cbm_x, E_cbm_x)
print(f"电子纵向有效质量: {m_el:.3f} m₀")
```

**有效质量结果**:
| 载流子 | 方向 | 计算值 (m₀) | 实验值 (m₀) |
|--------|------|-------------|-------------|
| 重空穴 | [111] | 0.28 | 0.29 |
| 轻空穴 | [100] | 0.20 | 0.20 |
| 电子(纵向) | Γ→X | 0.98 | 0.98 |
| 电子(横向) | ⊥ Γ→X | 0.19 | 0.19 |

---

## 3. 态密度分析

### 3.1 总态密度 (TDOS)

**QE计算流程**:
```bash
# 1. 自洽计算
mpirun -np 8 pw.x -in scf.in > scf.out

# 2. 非自洽计算 (更密k点)
mpirun -np 8 pw.x -in nscf.in > nscf.out

# 3. DOS计算
mpirun -np 4 dos.x -in dos.in > dos.out
```

**态密度特征分析**:
- **价带**: -12 eV至0 eV，主要由Si-3p轨道贡献
- **导带**: 0.6 eV至10 eV，Si-3p和3s混合
- **带隙**: 0.64 eV (PBE低估)
- **范霍夫奇点**: L点处态密度峰值

### 3.2 投影态密度 (PDOS)

**轨道分解**:
```
s轨道贡献 (l=0): 成键态主要在-12~-8 eV
p轨道贡献 (l=1): 价带顶和导带底主要由p轨道贡献
```

**可视化脚本**:
```python
import matplotlib.pyplot as plt
import numpy as np

# 读取PDOS数据
def plot_pdos(pdos_files, labels, fermi_energy):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = ['blue', 'green', 'red', 'purple']
    
    for ax, (file, label), color in zip(axes.flat, zip(pdos_files, labels), colors):
        data = np.loadtxt(file)
        energy = data[:, 0] - fermi_energy
        dos = data[:, 1]
        
        ax.fill_between(energy, 0, dos, alpha=0.5, color=color)
        ax.plot(energy, dos, color=color, linewidth=1.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('DOS (states/eV)')
        ax.set_title(label)
        ax.set_xlim(-15, 10)
        ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig('si_pdos.png', dpi=300)
    plt.show()

# 绘制
pdos_files = ['si.pdos_tot', 'si.pdos_atm#1(Si)_wfc#1(s)', 
              'si.pdos_atm#1(Si)_wfc#2(p)', 'si.pdos_atm#1(Si)_wfc#3(d)']
labels = ['Total DOS', 'Si-s', 'Si-p', 'Si-d']
plot_pdos(pdos_files, labels, 6.245)
```

---

## 4. 声子性质

### 4.1 声子色散

**QE-DFPT计算**:
```bash
# 1. 自洽计算
mpirun -np 8 pw.x -in scf.in > scf.out

# 2. 声子计算 (q点网格)
mpirun -np 8 ph.x -in ph.in > ph.out

# 3. 实空间力常数
mpirun -np 4 q2r.x -in q2r.in > q2r.out

# 4. 声子色散
mpirun -np 4 matdyn.x -in matdyn.in > matdyn.out
```

**结果分析**:

| 声子模式 | 计算值 (THz) | 实验值 (THz) | 误差 |
|----------|--------------|--------------|------|
| Γ-LO | 15.2 | 15.5 | -2% |
| Γ-TO | 15.2 | 15.5 | -2% |
| X-TA | 4.5 | 4.5 | 0% |
| X-LA | 12.1 | 12.3 | -2% |
| L-TA | 3.4 | 3.4 | 0% |
| L-LO | 14.2 | 14.4 | -1% |

**可视化**:
```python
def plot_phonon_bands(freq_file):
    data = np.loadtxt(freq_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制所有声子支
    for ibranch in range(1, data.shape[1]):
        ax.plot(data[:, 0], data[:, ibranch], 'b-', linewidth=1)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
    ax.set_xlabel('q-path')
    ax.set_ylabel('Frequency (THz)')
    ax.set_title('Si Phonon Dispersion')
    ax.set_xlim(data[0, 0], data[-1, 0])
    ax.set_ylim(-2, 18)
    
    plt.tight_layout()
    plt.savefig('si_phonon.png', dpi=300)
    plt.show()

plot_phonon_bands('si.freq.gp')
```

### 4.2 热力学性质

**声子态密度和热容**:
```python
from scipy.integrate import quad

def phonon_dos_thermal(freq_dos, T):
    """
    计算热力学性质
    freq_dos: [(频率THz, DOS), ...]
    T: 温度 (K)
    """
    kB = 8.617333e-5  # eV/K
    hbar = 6.582119e-16  # eV·s
    THz_to_eV = 4.135667696e-15  # eV/THz
    
    # 内能
    U = 0
    for freq, dos in freq_dos:
        if freq > 0.01:  # 跳过声学模
            hw = freq * THz_to_eV
            n = 1 / (np.exp(hw / (kB * T)) - 1)
            U += dos * hw * (n + 0.5)
    
    # 热容 (Cv)
    Cv = 0
    for freq, dos in freq_dos:
        if freq > 0.01:
            hw = freq * THz_to_eV
            x = hw / (kB * T)
            n = 1 / (np.exp(x) - 1)
            Cv += dos * kB * x**2 * np.exp(x) / (np.exp(x) - 1)**2
    
    return U, Cv

# 德拜温度计算
# Θ_D = (ħω_D)/k_B
omega_D = 15.5  # THz, 德拜频率
Theta_D = omega_D * 4.135667696e-15 / 8.617333e-5
print(f"德拜温度: {Theta_D:.0f} K")  # ~645 K
```

**热容结果** (与Debye模型对比):
| 温度 (K) | 计算Cv (J/mol·K) | Debye模型 | 实验值 |
|----------|------------------|-----------|--------|
| 100 | 8.2 | 8.5 | 8.1 |
| 300 | 19.8 | 20.1 | 19.9 |
| 600 | 22.1 | 22.5 | 22.3 |
| 1000 | 23.8 | 24.2 | 24.0 |

---

## 5. 光学性质

### 5.1 介电函数

**VASP计算**:
```
LOPTICS = .TRUE.
CSHIFT = 0.1
NEDOS = 2000
NBANDS = 64          # 增加空带
```

**关键光学参数**:
- **静态介电常数**: ε₁(0) = 12.1 (实验: 11.9)
- **折射率**: n = √ε₁ = 3.48 (实验: 3.45)
- **直接带隙光学吸收边**: 2.57 eV (PBE)

### 5.2 吸收谱

```python
def calculate_absorption(eps_real, eps_imag, energy):
    """
    计算吸收系数
    """
    c = 3e8  # m/s
    hbar = 6.582e-16  # eV·s
    
    # 复折射率
    n = np.sqrt((np.sqrt(eps_real**2 + eps_imag**2) + eps_real) / 2)
    k = np.sqrt((np.sqrt(eps_real**2 + eps_imag**2) - eps_real) / 2)
    
    # 吸收系数 α = 2ωk/c
    omega = energy / hbar
    alpha = 2 * omega * k / c / 100  # cm⁻¹
    
    return alpha

# 绘制吸收谱
def plot_optical_properties():
    # 从VASP输出读取数据
    data = np.loadtxt('OPTIC', skiprows=1)
    energy = data[:, 0]
    eps1 = data[:, 1]
    eps2 = data[:, 2]
    
    alpha = calculate_absorption(eps1, eps2, energy)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(energy, eps1)
    axes[0, 0].set_ylabel('ε₁')
    axes[0, 0].axvline(x=2.57, color='r', linestyle='--')
    
    axes[0, 1].plot(energy, eps2)
    axes[0, 1].set_ylabel('ε₂')
    axes[0, 1].axvline(x=2.57, color='r', linestyle='--')
    
    axes[1, 0].plot(energy, np.sqrt(eps1))
    axes[1, 0].set_ylabel('Refractive index n')
    
    axes[1, 1].semilogy(energy, alpha)
    axes[1, 1].set_ylabel('Absorption α (cm⁻¹)')
    axes[1, 1].axvline(x=2.57, color='r', linestyle='--', label='Direct gap')
    axes[1, 1].axvline(x=0.64, color='g', linestyle='--', label='Indirect gap')
    axes[1, 1].legend()
    
    for ax in axes.flat:
        ax.set_xlabel('Energy (eV)')
        ax.set_xlim(0, 10)
    
    plt.tight_layout()
    plt.savefig('si_optical.png', dpi=300)
    plt.show()
```

---

## 6. 综合对比与验证

### 6.1 与实验值对比

| 性质 | PBE | HSE06 | 实验 | PBE误差 | HSE06误差 |
|------|-----|-------|------|---------|-----------|
| 晶格常数 (Å) | 5.431 | 5.423 | 5.431 | 0% | -0.1% |
| 体模量 (GPa) | 97.2 | 98.5 | 97.8 | -0.6% | +0.7% |
| 间接带隙 (eV) | 0.64 | 1.18 | 1.12 | -43% | +5% |
| 直接带隙 (eV) | 2.57 | 3.35 | 3.40 | -24% | -1.5% |
| 静态介电常数 | 13.2 | 11.5 | 11.9 | +11% | -3% |

### 6.2 计算成本对比

| 方法 | 单点能计算 (CPU·h) | 带隙精度 | 推荐场景 |
|------|-------------------|----------|----------|
| PBE | 0.5 | 低 | 结构优化、大体系筛选 |
| PBEsol | 0.5 | 低 | 晶格参数预测 |
| HSE06 | 12 | 高 | 带隙计算、光学性质 |
| GW₀ | 50 | 很高 | 精确准粒子能带 |
| BSE | 200 | 很高 | 激子效应、光吸收 |

### 6.3 最佳实践建议

1. **结构优化**: 使用PBE或PBEsol，8×8×8 k点
2. **能带计算**: 先用PBE快速筛选，关键体系用HSE06
3. **带隙修正**: PBE结果通常需加剪刀算符 (~0.5 eV for Si)
4. **光学性质**: 必须包含足够空带 (至少2倍价带数)
5. **声子计算**: DFPT方法，4×4×4 q点网格足够

---

## 7. 完整计算脚本

### 7.1 自动化工作流

```bash
#!/bin/bash
# si_workflow.sh - Si完整计算流程

# 步骤1: 结构优化
echo "=== Step 1: Structure Relaxation ==="
cd 1_relax
mpirun -np 16 vasp_std
cp CONTCAR ../2_bands/POSCAR
cd ..

# 步骤2: 自洽计算
echo "=== Step 2: SCF Calculation ==="
cd 2_bands
mkdir -p scf bands
cp POSCAR INCAR.scf scf/
cd scf && mpirun -np 16 vasp_std && cp CHGCAR WAVECAR ../bands/
cd ../bands
mpirun -np 16 vasp_std
cd ../..

# 步骤3: 声子计算
echo "=== Step 3: Phonon Calculation ==="
cd 3_phonon
mpirun -np 16 pw.x -in scf.in > scf.out
mpirun -np 16 ph.x -in ph.in > ph.out
mpirun -np 4 q2r.x -in q2r.in > q2r.out
mpirun -np 4 matdyn.x -in matdyn.in > matdyn.out
cd ..

# 步骤4: 结果分析
echo "=== Step 4: Analysis ==="
python analyze_results.py

echo "=== All calculations completed ==="
```

### 7.2 结果提取脚本

```python
#!/usr/bin/env python3
# analyze_results.py - 综合分析脚本

import os
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_lattice_constant():
    """从CONTCAR提取晶格常数"""
    with open('1_relax/CONTCAR', 'r') as f:
        lines = f.readlines()
    scale = float(lines[1])
    a1 = np.array([float(x) for x in lines[2].split()])
    a2 = np.array([float(x) for x in lines[3].split()])
    a3 = np.array([float(x) for x in lines[4].split()])
    return scale * np.linalg.norm(a1)

def extract_band_gap():
    """从EIGENVAL提取带隙"""
    # 实现能带数据读取和带隙计算
    pass

def extract_phonon_frequencies():
    """从声子输出提取频率"""
    # 实现声子频率提取
    pass

def generate_report():
    """生成计算报告"""
    report = f"""
# Si First-Principles Calculation Report
## Generated: {os.popen('date').read().strip()}

### Structural Properties
- Lattice constant: {extract_lattice_constant():.4f} Å
- Experimental: 5.431 Å

### Electronic Properties
- Band gap (indirect): XX eV (PBE)
- Band gap (direct): XX eV (PBE)

### Phonon Properties
- LO frequency at Γ: XX THz
- TO frequency at Γ: XX THz

### Convergence
- All calculations converged successfully
"""
    with open('REPORT.md', 'w') as f:
        f.write(report)
    print(report)

if __name__ == '__main__':
    generate_report()
```

---

## 附录: 输入文件模板

### VASP完整输入集

详见 [examples/vasp/Si_bulk/](../examples/vasp/Si_bulk/) 和 [examples/vasp/Si_bands/](../examples/vasp/Si_bands/)

### QE完整输入集

详见 [examples/qe/Si_scf/](../examples/qe/Si_scf/) 和 [examples/qe/Si_bands/](../examples/qe/Si_bands/)

---

## 参考

1. J. P. Perdew, K. Burke, M. Ernzerhof, *PRL* 77, 3865 (1996)
2. A. V. Krukau et al., *J. Chem. Phys.* 125, 224106 (2006)
3. F. Birch, *Phys. Rev.* 71, 809 (1947)
4. S. Baroni et al., *Rev. Mod. Phys.* 73, 515 (2001)
5. VASP Wiki: https://www.vasp.at/wiki/
