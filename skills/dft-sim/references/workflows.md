# 多模块顺序工作流

本文档提供第一性原理计算中常见的多步骤工作流程，每个流程都包含详细的步骤说明、输入文件示例和自动化脚本。

---

## 1. 能带+有效质量计算流程

### 1.1 流程概述

```
步骤1: 结构优化 → 获得平衡结构
    ↓
步骤2: 自洽计算 → 获得收敛电荷密度
    ↓
步骤3: 能带计算 → 获得电子能带结构
    ↓
步骤4: 有效质量拟合 → 计算载流子有效质量
```

### 1.2 VASP实现

**完整脚本** (`workflow_bands_mass.sh`):
```bash
#!/bin/bash

SYSTEM="Si"
NCPU=16

echo "=========================================="
echo "Band Structure + Effective Mass Workflow"
echo "System: $SYSTEM"
echo "=========================================="

# 步骤1: 结构优化
echo "Step 1: Structure Optimization"
cd 01_relax || exit 1

cat > INCAR << EOF
SYSTEM = $SYSTEM Relaxation
ISMEAR = -5
ENCUT = 400
EDIFF = 1E-6

IBRION = 2
ISIF = 3
NSW = 100
EDIFFG = -0.01
EOF

mpirun -np $NCPU vasp_std
if [ ! -f CONTCAR ]; then
    echo "ERROR: Relaxation failed"
    exit 1
fi
cp CONTCAR ../02_scf/POSCAR
cd ..

# 步骤2: 自洽计算
echo "Step 2: SCF Calculation"
cd 02_scf || exit 1

cat > INCAR << EOF
SYSTEM = $SYSTEM SCF
ISMEAR = -5
ENCUT = 400
EDIFF = 1E-8
LORBIT = 11

LWAVE = .TRUE.
LCHARG = .TRUE.
EOF

cp ../01_relax/CONTCAR POSCAR
mpirun -np $NCPU vasp_std
cp WAVECAR CHGCAR ../03_bands/
cd ..

# 步骤3: 能带计算
echo "Step 3: Band Structure Calculation"
cd 03_bands || exit 1

cat > INCAR << EOF
SYSTEM = $SYSTEM Bands
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400
EDIFF = 1E-8

ISTART = 1
ICHARG = 11
LORBIT = 11
EOF

# 高对称k点路径 (Si)
cat > KPOINTS << EOF
k-points along high symmetry line
40
Line-mode
Reciprocal
0.0 0.0 0.0    ! Gamma
0.5 0.0 0.5    ! X

0.5 0.0 0.5    ! X
0.5 0.25 0.75  ! W

0.5 0.25 0.75  ! W
0.0 0.0 0.0    ! Gamma

0.0 0.0 0.0    ! Gamma
0.5 0.5 0.5    ! L
EOF

mpirun -np $NCPU vasp_std
cd ..

# 步骤4: 有效质量计算
echo "Step 4: Effective Mass Calculation"
cd 04_effective_mass || exit 1

# 使用sumo或自定义脚本
python3 calc_effective_mass.py ../03_bands/EIGENVAL

cd ..

echo "=========================================="
echo "Workflow completed successfully!"
echo "Results:"
echo "  - Optimized structure: 01_relax/CONTCAR"
echo "  - Band structure: 03_bands/EIGENVAL"
echo "  - Effective mass: 04_effective_mass/mass.dat"
echo "=========================================="
```

**有效质量计算脚本** (`calc_effective_mass.py`):
```python
#!/usr/bin/env python3
"""
计算有效质量
"""

import numpy as np
import sys
from scipy.optimize import curve_fit

def parse_eigenval(filename='EIGENVAL'):
    """解析VASP EIGENVAL文件"""
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 读取基本信息
    nkpoints = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    
    # 读取能带数据
    kpoints = []
    energies = []
    
    line_idx = 7
    for _ in range(nkpoints):
        k_coord = list(map(float, lines[line_idx].split()[:3]))
        kpoints.append(k_coord)
        
        band_energies = []
        for i in range(nbands):
            e = float(lines[line_idx + 1 + i].split()[1])
            band_energies.append(e)
        
        energies.append(band_energies)
        line_idx += nbands + 2
    
    return np.array(kpoints), np.array(energies)

def parabolic_fit(k, E0, m_eff):
    """抛物线拟合: E = E0 + hbar^2*k^2/(2*m_eff)"""
    hbar_sq = 0.076199682  # eV*Å²*amu (原子单位)
    return E0 + hbar_sq * k**2 / (2 * m_eff)

def calc_effective_mass(kpoints, energies, band_idx, k_range):
    """
    计算特定能带的有效质量
    
    k_range: (k_start, k_end) 在k空间中的范围
    """
    
    # 计算k点距离 (假设等间距)
    k_distances = np.zeros(len(kpoints))
    for i in range(1, len(kpoints)):
        dk = np.linalg.norm(kpoints[i] - kpoints[i-1])
        k_distances[i] = k_distances[i-1] + dk
    
    # 选择数据点
    mask = (k_distances >= k_range[0]) & (k_distances <= k_range[1])
    k_selected = k_distances[mask]
    e_selected = energies[mask, band_idx]
    
    # 找到极值点
    e_min = np.min(e_selected)
    e_max = np.max(e_selected)
    
    # 拟合
    if abs(e_min) < abs(e_max):  # 导带底
        k0 = k_selected[np.argmin(e_selected)]
        e0 = e_min
        popt, _ = curve_fit(parabolic_fit, k_selected - k0, e_selected)
        m_eff = popt[1]
    else:  # 价带顶
        k0 = k_selected[np.argmax(e_selected)]
        e0 = e_max
        popt, _ = curve_fit(parabolic_fit, k_selected - k0, -e_selected)
        m_eff = -popt[1]
    
    return m_eff, e0, k0

if __name__ == '__main__':
    eigenval_file = sys.argv[1] if len(sys.argv) > 1 else 'EIGENVAL'
    
    kpoints, energies = parse_eigenval(eigenval_file)
    
    print(f"Number of k-points: {len(kpoints)}")
    print(f"Number of bands: {energies.shape[1]}")
    
    # 示例: 计算导带底有效质量
    # 需要根据实际能带结构调整参数
    cbm_idx = energies.shape[1] // 2  # 假设导带底在中间
    k_range = (0, 0.5)  # k空间范围
    
    m_eff, e0, k0 = calc_effective_mass(kpoints, energies, cbm_idx, k_range)
    
    print(f"\nEffective mass calculation:")
    print(f"  Band index: {cbm_idx}")
    print(f"  Energy at extremum: {e0:.4f} eV")
    print(f"  Effective mass: {m_eff:.4f} m0")
    
    # 保存结果
    with open('mass.dat', 'w') as f:
        f.write(f"# Effective mass calculation\n")
        f.write(f"# Band index: {cbm_idx}\n")
        f.write(f"# Energy (eV): {e0:.6f}\n")
        f.write(f"Effective mass (m0): {m_eff:.6f}\n")
```

---

## 2. 声子+热力学计算流程

### 2.1 流程概述

```
步骤1: 结构优化 → 获得平衡结构
    ↓
步骤2: 声子计算 → 获得声子色散和DOS
    ↓
步骤3: 热力学性质 → 计算热容、熵等
    ↓
步骤4: 热膨胀 → 计算热膨胀系数
```

### 2.2 QE实现 (推荐)

**完整脚本** (`workflow_phonon.sh`):
```bash
#!/bin/bash

SYSTEM="Si"
NCPU=16

echo "=========================================="
echo "Phonon + Thermodynamics Workflow"
echo "System: $SYSTEM"
echo "=========================================="

# 步骤1: 结构优化
echo "Step 1: Structure Optimization"
cd 01_relax || exit 1

cat > relax.in << EOF
&CONTROL
  calculation = 'vc-relax'
  prefix = '$SYSTEM'
  outdir = './tmp'
  pseudo_dir = '../pseudo'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
/
&ELECTRONS
  conv_thr = 1.0d-10
/
&IONS
  ion_dynamics = 'bfgs'
/
&CELL
  cell_dynamics = 'bfgs'
  press_conv_thr = 0.5
/
ATOMIC_SPECIES
  Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (alat)
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS automatic
  8 8 8 0 0 0
EOF

mpirun -np $NCPU pw.x -in relax.in > relax.out
cp ${SYSTEM}.save ../02_scf/
cd ..

# 步骤2: SCF计算 (用于声子)
echo "Step 2: SCF Calculation for Phonon"
cd 02_scf || exit 1

cat > scf.in << EOF
&CONTROL
  calculation = 'scf'
  prefix = '$SYSTEM'
  outdir = './tmp'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
/
&ELECTRONS
  conv_thr = 1.0d-12
/
ATOMIC_SPECIES
  Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (alat)
  Si 0.00 0.00 0.00
  Si 0.25 0.25 0.25
K_POINTS automatic
  8 8 8 0 0 0
EOF

mpirun -np $NCPU pw.x -in scf.in > scf.out
cd ..

# 步骤3: 声子计算
echo "Step 3: Phonon Calculation"
cd 03_phonon || exit 1

cat > ph.in << EOF
Phonon calculation
&INPUTPH
  tr2_ph = 1.0d-14
  prefix = '$SYSTEM'
  outdir = '../02_scf/tmp'
  fildyn = '${SYSTEM}.dyn'
  ldisp = .true.
  nq1 = 4
  nq2 = 4
  nq3 = 4
  epsil = .true.
/
EOF

mpirun -np $NCPU ph.x -in ph.in > ph.out

# 后处理: q2r和matdyn
cat > q2r.in << EOF
&INPUT
  fildyn = '${SYSTEM}.dyn'
  flfrc = '${SYSTEM}.fc'
  la2F = .false.
/
EOF

mpirun -np 1 q2r.x -in q2r.in > q2r.out

# 声子色散
cat > matdyn_disp.in << EOF
&INPUT
  flfrc = '${SYSTEM}.fc'
  flfrq = '${SYSTEM}.freq'
  q_in_band_form = .true.
/
6
  0.0 0.0 0.0 20  ! Gamma
  0.5 0.0 0.5 20  ! X
  0.5 0.25 0.75 20 ! W
  0.0 0.0 0.0 20  ! Gamma
  0.5 0.5 0.5 20  ! L
  0.0 0.0 0.0 1   ! Gamma
EOF

mpirun -np 1 matdyn.x -in matdyn_disp.in > matdyn_disp.out

# 声子DOS
cat > matdyn_dos.in << EOF
&INPUT
  flfrc = '${SYSTEM}.fc'
  flfrq = '${SYSTEM}.freq.dos'
  dos = .true.
  fldos = '${SYSTEM}.phdos'
  nk1 = 20
  nk2 = 20
  nk3 = 20
/
EOF

mpirun -np 1 matdyn.x -in matdyn_dos.in > matdyn_dos.out

cd ..

# 步骤4: 热力学性质
echo "Step 4: Thermodynamic Properties"
cd 04_thermodynamics || exit 1

python3 calc_thermodynamics.py ../03_phonon/${SYSTEM}.phdos

cd ..

echo "=========================================="
echo "Workflow completed successfully!"
echo "=========================================="
```

**热力学计算脚本** (`calc_thermodynamics.py`):
```python
#!/usr/bin/env python3
"""
计算热力学性质
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

def parse_phdos(filename):
    """解析声子DOS文件"""
    data = np.loadtxt(filename)
    freq = data[:, 0]  # cm^-1
    dos = data[:, 1]
    return freq, dos

def calc_thermodynamics(freq, dos, T_range):
    """
    计算热力学性质
    
    使用声子DOS计算:
    - 热容 Cv
    - 熵 S
    - 自由能 F
    """
    
    # 常数
    kB = 8.617333e-5  # eV/K
    hbar = 4.135667e-15  # eV*s
    c = 2.998e10  # cm/s
    
    # 转换频率到能量 (eV)
    omega = 2 * np.pi * freq * c * hbar  # eV
    
    # 积分DOS
    dos_integral = np.trapz(dos, omega)
    
    results = {'T': [], 'Cv': [], 'S': [], 'F': []}
    
    for T in T_range:
        if T == 0:
            results['T'].append(T)
            results['Cv'].append(0)
            results['S'].append(0)
            results['F'].append(0)
            continue
        
        # 玻色-爱因斯坦分布
        x = omega / (kB * T)
        n_bose = 1 / (np.exp(x) - 1)
        
        # 热容 (eV/K per cell)
        integrand_cv = dos * kB * x**2 * np.exp(x) / (np.exp(x) - 1)**2
        cv = np.trapz(integrand_cv, omega)
        
        # 熵 (eV/K per cell)
        integrand_s = dos * kB * ((1 + n_bose) * np.log(1 + n_bose) - n_bose * np.log(n_bose))
        s = np.trapz(integrand_s, omega)
        
        # 自由能 (eV per cell)
        integrand_f = dos * kB * T * np.log(1 - np.exp(-x))
        f = np.trapz(integrand_f, omega)
        
        results['T'].append(T)
        results['Cv'].append(cv)
        results['S'].append(s)
        results['F'].append(f)
    
    return results

def plot_thermodynamics(results, output='thermodynamics.png'):
    """绘制热力学性质"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    T = np.array(results['T'])
    
    # 热容
    axes[0, 0].plot(T, results['Cv'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Temperature (K)')
    axes[0, 0].set_ylabel('Cv (eV/K per cell)')
    axes[0, 0].set_title('Heat Capacity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 熵
    axes[0, 1].plot(T, results['S'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Temperature (K)')
    axes[0, 1].set_ylabel('S (eV/K per cell)')
    axes[0, 1].set_title('Entropy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 自由能
    axes[1, 0].plot(T, results['F'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Temperature (K)')
    axes[1, 0].set_ylabel('F (eV per cell)')
    axes[1, 0].set_title('Helmholtz Free Energy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cv/T^3 (低温行为)
    cv_t3 = np.array(results['Cv']) / T**3
    axes[1, 1].plot(T[1:], cv_t3[1:], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Temperature (K)')
    axes[1, 1].set_ylabel('Cv/T^3 (eV/K^4 per cell)')
    axes[1, 1].set_title('Low Temperature Behavior')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Thermodynamics plot saved to {output}")

if __name__ == '__main__':
    phdos_file = sys.argv[1] if len(sys.argv) > 1 else 'phdos.out'
    
    freq, dos = parse_phdos(phdos_file)
    print(f"Phonon frequency range: {freq.min():.2f} to {freq.max():.2f} cm^-1")
    
    # 计算热力学性质 (0-1000K)
    T_range = np.linspace(0, 1000, 101)
    results = calc_thermodynamics(freq, dos, T_range)
    
    # 保存数据
    with open('thermodynamics.dat', 'w') as f:
        f.write("# T(K) Cv(eV/K) S(eV/K) F(eV)\n")
        for i in range(len(results['T'])):
            f.write(f"{results['T'][i]:.2f} {results['Cv'][i]:.6e} "
                   f"{results['S'][i]:.6e} {results['F'][i]:.6e}\n")
    
    print("Thermodynamics data saved to thermodynamics.dat")
    
    # 绘图
    plot_thermodynamics(results)
```

---

## 3. 缺陷完整流程

### 3.1 流程概述

```
步骤1: 体材料优化 → 获得平衡晶格常数
    ↓
步骤2: 超胞构建 → 创建足够大的超胞
    ↓
步骤3: 缺陷结构优化 → 优化缺陷周围原子位置
    ↓
步骤4: 形成能计算 → 计算不同电荷态的形成能
    ↓
步骤5: 跃迁能级 → 确定(0/+)和(+/-)等跃迁能级
```

### 3.2 自动化脚本

**完整脚本** (`workflow_defect.sh`):
```bash
#!/bin/bash

DEFECT="V_Si"
SUPERCELL_SIZE=3
NCPU=16

echo "=========================================="
echo "Defect Calculation Workflow"
echo "Defect: $DEFECT"
echo "Supercell: ${SUPERCELL_SIZE}x${SUPERCELL_SIZE}x${SUPERCELL_SIZE}"
echo "=========================================="

# 步骤1: 体材料优化
echo "Step 1: Bulk Optimization"
cd 01_bulk_opt || exit 1
# ... (VASP或QE计算)
cd ..

# 步骤2: 构建超胞
echo "Step 2: Build Supercell"
cd 02_supercell || exit 1
python3 << EOF
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.defects.core import Vacancy

bulk = Structure.from_file('../01_bulk_opt/CONTCAR')
supercell = bulk * [$SUPERCELL_SIZE, $SUPERCELL_SIZE, $SUPERCELL_SIZE]

# 创建空位
vacancy = Vacancy(supercell, supercell[0], charge=0)
defect_structure = vacancy.generate_defect_structure()

Poscar(defect_structure).write_file('POSCAR_defect')
Poscar(supercell).write_file('POSCAR_bulk_supercell')
EOF
cd ..

# 步骤3: 缺陷结构优化
echo "Step 3: Defect Structure Relaxation"
cd 03_defect_relax || exit 1
cp ../02_supercell/POSCAR_defect POSCAR
# ... (VASP计算)
cd ..

# 步骤4: 不同电荷态计算
echo "Step 4: Charge State Calculations"
for q in -2 -1 0 1 2; do
    echo "  Calculating q=$q"
    cd 04_charge_q${q} || continue
    cp ../03_defect_relax/CONTCAR POSCAR
    # 修改INCAR设置NELECT
    # ... (VASP计算)
    cd ..
done

# 步骤5: 分析
echo "Step 5: Analysis"
cd 05_analysis || exit 1
python3 analyze_defect.py
cd ..

echo "=========================================="
echo "Defect workflow completed!"
echo "=========================================="
```

---

## 4. 电催化计算流程

### 4.1 流程概述

```
步骤1: 表面优化 → 获得稳定的催化剂表面
    ↓
步骤2: 吸附能计算 → 计算反应中间体吸附能
    ↓
步骤3: 自由能修正 → 添加零点能和熵修正
    ↓
步骤4: Pourbaix图 → 构建电化学稳定性相图
```

### 4.2 关键计算

**吸附能计算**:
```python
#!/usr/bin/env python3
"""
计算吸附能
E_ads = E_surface+adsorbate - E_surface - E_adsorbate
"""

def calc_adsorption_energy(
    e_surface_ads,    # 吸附体系的能量
    e_surface,        # 清洁表面的能量
    e_adsorbate,      # 气相吸附物的能量
    n_adsorbate=1     # 吸附物数量
):
    """计算吸附能"""
    e_ads = e_surface_ads - e_surface - n_adsorbate * e_adsorbate
    return e_ads

# 示例
E_OH = calc_adsorption_energy(-450.2, -400.0, -50.0)
print(f"OH adsorption energy: {E_OH:.3f} eV")
```

**自由能修正**:
```python
#!/usr/bin/env python3
"""
计算电化学自由能
G = E_DFT + ZPE - TS + ∫ Cp dT
"""

def calc_free_energy(
    e_dft,           # DFT能量
    zpe,             # 零点能
    entropy,         # 熵 (eV/K)
    temperature=298.15
):
    """计算自由能"""
    g = e_dft + zpe - temperature * entropy
    return g

# 自由能修正 (简化)
# H2O (g): ZPE = 0.56 eV, S = 0.002 eV/K
# H2 (g): ZPE = 0.27 eV, S = 0.0014 eV/K
# OH*: ZPE = 0.35 eV, S = 0.0005 eV/K (表面吸附，熵较小)
```

---

## 5. 电池材料计算流程

### 5.1 流程概述

```
步骤1: 开路电压计算 → 计算不同锂化状态的电压
    ↓
步骤2: 离子迁移势垒 → NEB计算离子扩散路径
    ↓
步骤3: 相稳定性 → 构建凸包图分析稳定性
```

### 5.2 开路电压计算

```python
#!/usr/bin/env python3
"""
计算电池开路电压
V = -[E(Li_x2) - E(Li_x1) - (x2-x1)*E(Li)] / (x2-x1) / e
"""

def calc_ocv(
    e_cathode_x1,    # Li_x1阴极能量
    e_cathode_x2,    # Li_x2阴极能量
    e_li_metal,      # 金属锂能量
    x1, x2           # 锂化程度
):
    """计算平均开路电压"""
    voltage = -(e_cathode_x2 - e_cathode_x1 - (x2-x1)*e_li_metal) / (x2-x1)
    return voltage

# 示例: LiCoO2
E_LiCoO2 = -250.0
E_CoO2 = -200.0
E_Li = -5.0

V_ocv = calc_ocv(E_LiCoO2, E_CoO2, E_Li, 1.0, 0.0)
print(f"OCV: {V_ocv:.3f} V")
```

---

## 6. 参考文献

1. ** workflow automation**: https://materialsproject.org/
2. **ASE workflows**: https://wiki.fysik.dtu.dk/ase/
3. **AiiDA workflows**: https://www.aiida.net/
4. **Atomate workflows**: https://github.com/hackingmaterials/atomate

---

*文档版本: 1.0*
*更新日期: 2026-03-08*
