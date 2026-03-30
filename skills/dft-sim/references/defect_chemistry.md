# 缺陷化学计算方法详解

## 1. 原理简述

### 1.1 点缺陷类型

| 缺陷类型 | 符号 | 描述 |
|---------|------|------|
| **空位** | V_X | X原子缺失 |
| **间隙原子** | X_i | X原子位于间隙位置 |
| **替位原子** | Y_X | Y原子替代X原子位置 |
| **反位缺陷** | X_Y | X原子占据Y原子位置 |
| **弗伦克尔对** | (V_X + X_i) | 空位+间隙对 |

### 1.2 缺陷形成能

缺陷形成能计算公式:
```
E_form = E_defect - E_bulk - Σ n_i μ_i + q(E_F + E_VBM) + E_corr
```

**各项含义**:
- **E_defect**: 含缺陷超胞总能量
- **E_bulk**: 完美晶体超胞总能量
- **n_i**: 添加(n_i>0)或移除(n_i<0)的原子数
- **μ_i**: 原子化学势
- **q**: 缺陷电荷态
- **E_F**: 费米能级 (相对于VBM)
- **E_VBM**: 价带顶能量
- **E_corr**: 有限尺寸修正

### 1.3 跃迁能级 (Transition Levels)

跃迁能级 ε(q/q') 是缺陷从电荷态q变为q'时的费米能级位置:
```
ε(q/q') = [E_defect(q) - E_defect(q')] / (q' - q)
```

### 1.4 有限尺寸修正

| 修正方法 | 适用情况 | 公式 |
|---------|---------|------|
| **Makov-Payne** | 立方晶胞 | E_corr = 2/3 × q²α/L |
| **Freysoldt** | 各向异性 | 势能对齐法 |
| **FNV** | 带电缺陷 | 结合势能对齐和图像电荷 |
| **Kumagai-Oba** | 各向异性 | 扩展FNV方法 |

---

## 2. VASP实现

### 2.1 缺陷超胞构建

**步骤1: 确定超胞大小**
```python
#!/usr/bin/env python3
"""构建缺陷超胞"""

from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.defects.core import Vacancy, Interstitial, Substitution

# 读取体结构
bulk = Structure.from_file("POSCAR_bulk")

# 创建3×3×3超胞
supercell = bulk * [3, 3, 3]

# 创建空位 (例如Si空位)
vacancy = Vacancy(supercell, supercell[0], charge=0)
defect_structure = vacancy.generate_defect_structure()

# 保存
Poscar(defect_structure).write_file("POSCAR_V_Si")

# 创建带电缺陷 (q = +2)
vacancy_q2 = Vacancy(supercell, supercell[0], charge=2)
defect_q2_structure = vacancy_q2.generate_defect_structure()
Poscar(defect_q2_structure).write_file("POSCAR_V_Si_q+2")
```

### 2.2 中性缺陷计算

**INCAR**:
```
SYSTEM = Si Vacancy (neutral)
ISMEAR = -5
ENCUT = 400
EDIFF = 1E-6

# 结构优化
IBRION = 2
ISIF = 2        # 优化离子位置，固定晶胞
NSW = 100
EDIFFG = -0.01

# 输出
LCHARG = .TRUE.
LWAVE = .TRUE.
```

### 2.3 带电缺陷计算

**关键设置**:
```
SYSTEM = Si Vacancy (q=+2)
ISMEAR = -5
ENCUT = 400
EDIFF = 1E-6

# 带电缺陷
NELECT = 214    # 调整电子数 (超胞总电子数 - 2)

# 优化
IBRION = 2
ISIF = 2
NSW = 100

# 偶极修正 (FNV修正)
IDIPOL = 4      # 4=全方向平均
LDIPOL = .TRUE.
```

### 2.4 缺陷形成能计算

**Python脚本** (`calc_formation_energy.py`):
```python
#!/usr/bin/env python3
"""
计算缺陷形成能
"""

import numpy as np

def parse_oszicar(filename='OSZICAR'):
    """从OSZICAR提取能量"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in reversed(lines):
        if 'TOTEN' in line:
            energy = float(line.split()[4])
            return energy
    return None

def calc_formation_energy(
    e_defect,           # 缺陷超胞能量
    e_bulk,             # 体材料超胞能量
    n_atoms,            # 添加(+)或移除(-)的原子数
    chem_potential,     # 原子化学势
    charge,             # 缺陷电荷态
    e_vbm,              # 价带顶能量
    e_fermi,            # 费米能级 (相对于VBM)
    e_corr=0.0          # 有限尺寸修正
):
    """
    计算缺陷形成能
    
    E_form = E_defect - E_bulk - n*μ + q*(E_F + E_VBM) + E_corr
    """
    
    formation_energy = (
        e_defect 
        - e_bulk 
        - n_atoms * chem_potential 
        + charge * (e_fermi + e_vbm)
        + e_corr
    )
    
    return formation_energy

def freysoldt_correction(
    charge,
    dielectric_constant,
    lattice_vectors,
    defect_position,
    potential_file='LOCPOT'
):
    """
    Freysoldt有限尺寸修正
    
    简化版本 - 实际需要更复杂的实现
    """
    # 简化的势能对齐修正
    # 实际实现需要读取LOCPOT并进行势能分析
    
    # 图像电荷修正 (Makov-Payne)
    L = np.mean([np.linalg.norm(v) for v in lattice_vectors])
    alpha = 2.8373  # Madelung常数 (立方晶胞)
    
    e_image = - (2/3) * charge**2 * alpha / (dielectric_constant * L)
    
    # 势能对齐 (简化)
    e_align = 0.0   # 需要从势能分析获得
    
    return e_image + e_align

if __name__ == '__main__':
    # 示例: Si空位形成能计算
    
    # 读取能量
    e_defect = parse_oszicar('defect/OSZICAR')
    e_bulk = parse_oszicar('bulk/OSZICAR')
    
    # 参数设置
    n_si = -1           # 移除1个Si原子
    mu_si = -5.4        # Si化学势 (eV)
    charge = 0          # 中性缺陷
    e_vbm = 0.0         # VBM能量 (相对于自身)
    e_fermi = 0.5       # 费米能级 (eV，相对于VBM)
    
    # 计算
    e_form = calc_formation_energy(
        e_defect, e_bulk, n_si, mu_si,
        charge, e_vbm, e_fermi
    )
    
    print(f"Defect formation energy: {e_form:.3f} eV")
    
    # 带电缺陷 (q = +2)
    e_defect_q2 = parse_oszicar('defect_q+2/OSZICAR')
    charge_q2 = 2
    
    e_form_q2 = calc_formation_energy(
        e_defect_q2, e_bulk, n_si, mu_si,
        charge_q2, e_vbm, e_fermi
    )
    
    print(f"Defect formation energy (q=+2): {e_form_q2:.3f} eV")
```

### 2.5 化学势相图

**化学势计算**:
```python
#!/usr/bin/env python3
"""
计算化学势相图
"""

import numpy as np
import matplotlib.pyplot as plt

def calc_chemical_potential(
    e_compound,      # 化合物总能量
    e_elements,      # 各元素体能量字典 {'Si': -5.4, 'O': -4.9}
    stoichiometry    # 化学计量比字典 {'Si': 1, 'O': 2}
):
    """
    计算化学势范围
    
    对于SiO2:
    μ_Si + 2μ_O = E(SiO2)
    μ_Si ≤ E(Si_bulk)
    μ_O ≤ E(O2)/2
    """
    
    # 稳定条件
    total = sum(stoichiometry.values())
    
    # 计算化学势范围
    mu_ranges = {}
    
    for elem in stoichiometry:
        # 富该元素极限
        mu_max = e_elements[elem]
        
        # 贫该元素极限
        others_sum = sum(
            stoichiometry[e] * e_elements[e] 
            for e in stoichiometry if e != elem
        )
        mu_min = e_compound - others_sum
        
        mu_ranges[elem] = (mu_min, mu_max)
    
    return mu_ranges

def plot_formation_energy_vs_fermi(
    transition_levels,
    formation_energies,
    band_gap,
    output='formation_energy.png'
):
    """
    绘制形成能随费米能级变化图
    
    transition_levels: [(e1, q1, q2), (e2, q3, q4), ...]
    formation_energies: 不同电荷态的形成能参数
    """
    
    e_fermi = np.linspace(0, band_gap, 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算各电荷态的形成能
    for charge, params in formation_energies.items():
        e_form = params['intercept'] + charge * e_fermi
        ax.plot(e_fermi, e_form, label=f'q={charge:+d}')
    
    # 标记跃迁能级
    for level, q1, q2 in transition_levels:
        ax.axvline(x=level, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'ε({q1}/{q2})', xy=(level, ax.get_ylim()[1]*0.9),
                   rotation=90, fontsize=9)
    
    ax.set_xlabel('Fermi Level (eV)', fontsize=12)
    ax.set_ylabel('Formation Energy (eV)', fontsize=12)
    ax.set_title('Defect Formation Energy vs Fermi Level', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, band_gap)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Plot saved to {output}")

if __name__ == '__main__':
    # 示例: SiO2化学势
    e_sio2 = -20.0
    e_elements = {'Si': -5.4, 'O': -4.9}
    stoichiometry = {'Si': 1, 'O': 2}
    
    mu_ranges = calc_chemical_potential(e_sio2, e_elements, stoichiometry)
    
    print("Chemical potential ranges:")
    for elem, (mu_min, mu_max) in mu_ranges.items():
        print(f"  μ_{elem}: {mu_min:.3f} to {mu_max:.3f} eV")
```

---

## 3. Quantum ESPRESSO实现

### 3.1 缺陷超胞计算

**scf.in**:
```fortran
&CONTROL
  calculation = 'scf'
  prefix = 'si_defect'
  outdir = './tmp'
/
&SYSTEM
  ibrav = 0
  nat = 215           # 3×3×3超胞 - 1个空位
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0d-10
/
CELL_PARAMETERS angstrom
  16.380000000   0.000000000   0.000000000
   0.000000000  16.380000000   0.000000000
   0.000000000   0.000000000  16.380000000
ATOMIC_SPECIES
  Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS angstrom
  Si   0.000000000   0.000000000   0.000000000
  ...
K_POINTS automatic
  2 2 2 0 0 0         # Γ点近似用于大超胞
```

### 3.2 带电缺陷

**带电缺陷设置**:
```fortran
&SYSTEM
  ...
  tot_charge = 2      # 总电荷 +2
/
```

### 3.3 缺陷形成能计算

**Python脚本** (`qe_defect_analysis.py`):
```python
#!/usr/bin/env python3
"""
分析QE缺陷计算结果
"""

import re
import numpy as np

def extract_energy_qe(output_file='pw.out'):
    """从QE输出提取总能量"""
    
    with open(output_file, 'r') as f:
        content = f.read()
    
    # 查找总能量
    match = re.search(r'total energy\s+=\s+([-\d.]+)\s+Ry', content)
    if match:
        energy_ry = float(match.group(1))
        energy_ev = energy_ry * 13.605698  # Ry to eV
        return energy_ev
    
    return None

def extract_vbm_qe(output_file='pw.out'):
    """提取价带顶能量"""
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    # 查找最高占据能级
    for line in reversed(lines):
        if 'highest occupied' in line.lower():
            match = re.search(r'([-\d.]+)\s*ev', line.lower())
            if match:
                return float(match.group(1))
    
    return None

if __name__ == '__main__':
    # 提取能量
    e_defect = extract_energy_qe('defect/pw.out')
    e_bulk = extract_energy_qe('bulk/pw.out')
    e_vbm = extract_vbm_qe('bulk/pw.out')
    
    print(f"Defect energy: {e_defect:.3f} eV")
    print(f"Bulk energy: {e_bulk:.3f} eV")
    print(f"VBM: {e_vbm:.3f} eV")
```

---

## 4. 完整工作流程

### 4.1 缺陷计算完整流程

```bash
#!/bin/bash
# run_defect_workflow.sh

# 1. 体材料优化
cd 01_bulk
mpirun -np 16 vasp_std
cd ..

# 2. 构建超胞和缺陷
python3 build_defect_supercell.py

# 3. 中性缺陷优化
cd 02_defect_neutral
cp ../01_bulk/WAVECAR .
mpirun -np 16 vasp_std
cd ..

# 4. 带电缺陷计算 (多个电荷态)
for q in -2 -1 0 1 2; do
    cd 03_defect_q${q}
    cp ../02_defect_neutral/CONTCAR POSCAR
    # 修改NELECT
    mpirun -np 16 vasp_std
    cd ..
done

# 5. 分析结果
python3 analyze_defects.py
python3 plot_formation_energy.py
```

### 4.2 分析脚本

**完整分析脚本** (`analyze_defects.py`):
```python
#!/usr/bin/env python3
"""
完整缺陷分析脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def analyze_defect_system(
    defect_name,
    bulk_energy,
    defect_energies,    # {charge: energy}
    vbm_energy,
    band_gap,
    chem_potential,
    n_atoms_removed,
    dielectric_const,
    lattice_constant
):
    """
    完整缺陷分析
    """
    
    print(f"\n{'='*60}")
    print(f"Defect Analysis: {defect_name}")
    print(f"{'='*60}")
    
    # 计算形成能
    formation_energies = {}
    
    for charge, e_def in defect_energies.items():
        # 简化修正 (实际需要更精确计算)
        corr = 0.0 if charge == 0 else -0.1 * charge**2
        
        e_form_0 = e_def - bulk_energy - n_atoms_removed * chem_potential + corr
        formation_energies[charge] = {
            'intercept': e_form_0,
            'slope': charge
        }
    
    # 计算跃迁能级
    charges = sorted(defect_energies.keys())
    transition_levels = []
    
    for i in range(len(charges)-1):
        q1, q2 = charges[i], charges[i+1]
        
        # 求解 E_form(q1) = E_form(q2)
        # intercept1 + q1*E_F = intercept2 + q2*E_F
        # E_F = (intercept2 - intercept1) / (q1 - q2)
        
        e_f = (formation_energies[q2]['intercept'] - 
               formation_energies[q1]['intercept']) / (q1 - q2)
        
        if 0 <= e_f <= band_gap:
            transition_levels.append((e_f, q1, q2))
            print(f"Transition level ε({q1:+d}/{q2:+d}) = {e_f:.3f} eV")
    
    # 绘制形成能图
    plot_formation_energy(
        formation_energies,
        transition_levels,
        band_gap,
        f'formation_energy_{defect_name}.png'
    )
    
    return formation_energies, transition_levels

def plot_formation_energy(formation_energies, transition_levels, band_gap, output):
    """绘制形成能图"""
    
    e_fermi = np.linspace(0, band_gap, 200)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算各电荷态形成能
    energies_at_ef = {q: [] for q in formation_energies}
    
    for ef in e_fermi:
        for q, params in formation_energies.items():
            e = params['intercept'] + params['slope'] * ef
            energies_at_ef[q].append(e)
    
    # 找到最低能量线
    min_energy = np.minimum.reduce(list(energies_at_ef.values()))
    ax.fill_between(e_fermi, min_energy, alpha=0.3)
    ax.plot(e_fermi, min_energy, 'k-', linewidth=2)
    
    # 绘制各电荷态
    for q, energies in energies_at_ef.items():
        ax.plot(e_fermi, energies, '--', alpha=0.5, label=f'q={q:+d}')
    
    # 标记跃迁能级
    for level, q1, q2 in transition_levels:
        ax.axvline(x=level, color='red', linestyle=':', alpha=0.7)
        ax.annotate(f'ε({q1:+d}/{q2:+d})', 
                   xy=(level, ax.get_ylim()[1]*0.9),
                   rotation=90, color='red', fontsize=10)
    
    ax.set_xlabel('Fermi Level (eV)', fontsize=12)
    ax.set_ylabel('Formation Energy (eV)', fontsize=12)
    ax.set_title('Defect Formation Energy', fontsize=14)
    ax.set_xlim(0, band_gap)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Plot saved to {output}")

if __name__ == '__main__':
    # 示例数据
    analyze_defect_system(
        defect_name='V_Si',
        bulk_energy=-1300.0,
        defect_energies={0: -1294.5, 1: -1290.0, 2: -1285.0, -1: -1298.0},
        vbm_energy=0.0,
        band_gap=1.12,
        chem_potential=-5.4,
        n_atoms_removed=1,
        dielectric_const=11.7,
        lattice_constant=5.43
    )
```

---

## 5. 常见错误和解决方案

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 超胞尺寸不足 | 缺陷-缺陷相互作用 | 增加超胞尺寸至收敛 |
| 带电缺陷不收敛 | 电荷分布问题 | 使用更密的k点或增加真空层 |
| 跃迁能级不合理 | 修正计算错误 | 检查势能对齐和图像电荷修正 |
| 形成能为负 | 化学势设置错误 | 检查化学势参考状态 |
| 结构优化失败 | 初始猜测差 | 使用更好的初始结构 |

---

## 6. 参考文献

1. **Freysoldt, C., et al. (2009)**. Fully ab initio finite-size corrections for charged-defect supercell calculations. *Physical Review Letters*, 102(1), 016402.

2. **Kumagai, Y., & Oba, F. (2014)**. Electrostatics-based finite-size corrections for first-principles point defect calculations. *Physical Review B*, 89(19), 195205.

3. **Northrup, J. E., & Zhang, S. B. (2009)**. Chemical potential dependence of defect formation energies in GaAs: Application to Ga self-diffusion. *Physical Review B*, 79(7), 073201.

4. **Freysoldt, C., et al. (2014)**. First-principles calculations for point defects in solids. *Reviews of Modern Physics*, 86(1), 253.

---

*文档版本: 1.0*
*更新日期: 2026-03-08*
