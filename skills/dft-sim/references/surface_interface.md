# 表面与界面计算方法详解

## 1. 原理简述

### 1.1 表面模型

表面计算通常使用**平板模型(Slab Model)**:
- 在xy平面保持周期性
- z方向添加足够厚的真空层 (~15-20 Å)
- 平板厚度需收敛测试

### 1.2 关键物理量

| 物理量 | 定义 | 公式 |
|--------|------|------|
| **表面能** | 创建单位面积表面所需能量 | γ = (E_slab - N×E_bulk)/(2A) |
| **功函数** | 电子从表面逃逸到真空所需能量 | Φ = E_vac - E_F |
| **粘附功** | 分离界面所需能量 | W_adh = E_surf1 + E_surf2 - E_interface |
| **表面偶极** | 表面电荷分布产生的偶极矩 | p = ∫ z·ρ(z) dz |

---

## 2. VASP实现

### 2.1 表面能计算

**步骤1: 体材料优化**
```
# INCAR
SYSTEM = Bulk Si
ISMEAR = -5
ENCUT = 400
EDIFF = 1E-6
IBRION = 2
ISIF = 3
NSW = 100
```

**步骤2: 构建Slab模型**
使用pymatgen或ASE:
```python
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

# 读取体结构
bulk = Structure.from_file("POSCAR_bulk")

# 创建表面 (111)面，3层原子
from pymatgen.core.surface import SlabGenerator
slab_gen = SlabGenerator(bulk, (1,1,1), min_slab_size=8, min_vacuum_size=20)
slabs = slab_gen.get_slabs()
slab = slabs[0]

# 保存
Poscar(slab).write_file("POSCAR_slab")
```

**步骤3: Slab优化**
```
# INCAR
SYSTEM = Si(111) Surface
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400
EDIFF = 1E-6

# 优化设置
IBRION = 2
ISIF = 0        # 只优化原子位置，固定晶胞
NSW = 100

# 选择性动力学 (可选)
# 固定底层原子
```

**POSCAR选择性动力学示例**:
```
Si(111) surface
1.0
   3.8400000000    0.0000000000    0.0000000000
   1.9200000000    3.3257100000    0.0000000000
   0.0000000000    0.0000000000   30.0000000000
  12
Direct
  0.0000000000  0.0000000000  0.1000000000  T T T
  0.3333333333  0.6666666667  0.1500000000  T T T
  ...
  0.0000000000  0.0000000000  0.0500000000  F F F  # 固定底层
```

**步骤4: 计算表面能**
```python
# surface_energy.py
E_slab = -120.5      # eV, 从OSZICAR获取
E_bulk_per_atom = -5.4  # eV
n_atoms = 12
area = 12.5          # Å² (根据晶胞计算)

gamma = (E_slab - n_atoms * E_bulk_per_atom) / (2 * area)
print(f"Surface energy: {gamma*16:.3f} J/m²")
print(f"Surface energy: {gamma*1.602:.3f} eV/Å²")
```

### 2.2 功函数计算

**关键设置**:
```
# INCAR
SYSTEM = Si(111) Work Function
ISMEAR = 0
SIGMA = 0.05
ENCUT = 400

# 功函数计算需要
LVHAR = .TRUE.      # 输出静电势
LVTOT = .TRUE.      # 输出总势

# 偶极修正 (对于不对称slab)
IDIPOL = 3          # z方向偶极修正
LDIPOL = .TRUE.
```

**提取功函数**:
```python
#!/usr/bin/env python3
"""提取VASP功函数"""

import numpy as np

def extract_workfunction(LOCPOT_file='LOCPOT'):
    """从LOCPOT提取功函数"""
    
    # 读取LOCPOT文件
    with open(LOCPOT_file, 'r') as f:
        lines = f.readlines()
    
    # 解析格点信息
    nx, ny, nz = map(int, lines[5].split())
    
    # 读取静电势数据
    potential = []
    for line in lines[6:]:
        potential.extend(map(float, line.split()))
    
    potential = np.array(potential)
    
    # 沿z方向平均
    v_z = potential.reshape(nx, ny, nz).mean(axis=(0, 1))
    
    # 真空能级 (取真空区平均值)
    vacuum_level = np.mean(v_z[-10:])  # 最后10个点
    
    # 费米能级 (从OUTCAR读取)
    with open('OUTCAR', 'r') as f:
        for line in f:
            if 'E-fermi' in line:
                e_fermi = float(line.split()[2])
                break
    
    workfunction = vacuum_level - e_fermi
    
    return workfunction, vacuum_level, e_fermi, v_z

if __name__ == '__main__':
    phi, v_vac, e_f, v_z = extract_workfunction()
    print(f"Vacuum level: {v_vac:.3f} eV")
    print(f"Fermi level: {e_f:.3f} eV")
    print(f"Work function: {phi:.3f} eV")
```

### 2.3 界面粘附功计算

**计算步骤**:
```python
#!/usr/bin/env python3
"""计算界面粘附功"""

# 能量 (从VASP获取)
E_interface = -250.0    # 界面体系总能量
E_surf1 = -120.0        # 表面1能量
E_surf2 = -130.0        # 表面2能量

# 粘附功
W_adh = E_surf1 + E_surf2 - E_interface
print(f"Adhesion work: {W_adh:.3f} eV")

# 界面能
gamma_interface = (E_interface - E_surf1 - E_surf2) / (2 * area)
print(f"Interface energy: {gamma_interface:.3f} eV/Å²")
```

### 2.4 关键参数详解

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `LVHAR` | 输出Hartree势 | .TRUE. |
| `LVTOT` | 输出总静电势 | .TRUE. |
| `IDIPOL` | 偶极修正方向 | 1/2/3 (x/y/z) |
| `LDIPOL` | 启用偶极修正 | .TRUE. |
| `EPSILON` | 介电常数 | 材料依赖 |
| `LVTOT` | 保存静电势 | .TRUE. |

---

## 3. Quantum ESPRESSO实现

### 3.1 表面能计算

**scf.in**:
```fortran
&CONTROL
  calculation = 'scf'
  prefix = 'si_slab'
  outdir = './tmp'
/
&SYSTEM
  ibrav = 0
  nat = 12
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0d-8
/
CELL_PARAMETERS angstrom
   3.840000000   0.000000000   0.000000000
   1.920000000   3.325710000   0.000000000
   0.000000000   0.000000000  30.000000000
ATOMIC_SPECIES
  Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS angstrom
  Si   0.000000000   0.000000000   3.000000000
  Si   1.920000000   1.108570000   4.500000000
  ...
K_POINTS automatic
  6 6 1 0 0 0
```

### 3.2 功函数计算 (pp.x)

**pp.in** (计算静电势):
```fortran
&INPUTPP
  prefix = 'si_slab'
  outdir = './tmp'
  filplot = 'potential.dat'
  plot_num = 11       # 静电势
/
&PLOT
  nfile = 1
  filepp(1) = 'potential.dat'
  weight(1) = 1.0
  iflag = 1           # 1D plot
  x0(1) = 0.0
  x0(2) = 0.0
  x0(3) = 0.0
  e1(1) = 0.0
  e1(2) = 0.0
  e1(3) = 1.0         # z方向
  nx = 100
  fileout = 'potential_1D.dat'
  output_format = 0
/
```

**运行**:
```bash
mpirun -np 8 pp.x -in pp.in > pp.out
```

**提取功函数**:
```python
#!/usr/bin/env python3
"""提取QE功函数"""

import numpy as np

def extract_qe_workfunction(potential_file='potential_1D.dat'):
    """从QE pp.x输出提取功函数"""
    
    # 读取数据
    data = np.loadtxt(potential_file)
    z = data[:, 0]      # z坐标
    v = data[:, 1]      # 静电势
    
    # 真空能级 (取最大值，通常在真空区)
    vacuum_level = np.max(v)
    
    # 费米能级 (从输出读取或设置)
    e_fermi = 5.0       # 从pw.x输出获取
    
    workfunction = vacuum_level - e_fermi
    
    return workfunction, vacuum_level, e_fermi, z, v

if __name__ == '__main__':
    phi, v_vac, e_f, z, v = extract_qe_workfunction()
    print(f"Vacuum level: {v_vac:.3f} eV")
    print(f"Fermi level: {e_f:.3f} eV")
    print(f"Work function: {phi:.3f} eV")
```

---

## 4. 完整输入文件示例

### 4.1 VASP: 金属表面功函数

**INCAR**:
```
SYSTEM = Al(111) Surface
ISMEAR = 1          # Methfessel-Paxton，用于金属
SIGMA = 0.2
ENCUT = 400
EDIFF = 1E-8

# 优化
IBRION = 2
ISIF = 0
NSW = 50
EDIFFG = -0.01

# 功函数计算
LVHAR = .TRUE.
LVTOT = .TRUE.

# 偶极修正 (不对称slab)
IDIPOL = 3
LDIPOL = .TRUE.

# 磁性 (如果需要)
ISPIN = 1
```

### 4.2 QE: 半导体表面

**完整脚本** (`run_surface.sh`):
```bash
#!/bin/bash

# 1. 体材料优化
mpirun -np 8 pw.x -in bulk_scf.in > bulk_scf.out
mpirun -np 8 pw.x -in bulk_relax.in > bulk_relax.out

# 2. 表面计算
mpirun -np 8 pw.x -in slab_scf.in > slab_scf.out
mpirun -np 8 pw.x -in slab_relax.in > slab_relax.out

# 3. 功函数计算
mpirun -np 8 pp.x -in potential.in > potential.out

# 4. 分析
echo "Surface energy calculation:"
python3 calc_surface_energy.py

echo "Work function calculation:"
python3 extract_workfunction.py
```

---

## 5. 常见错误和解决方案

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 表面能不收敛 | 平板厚度不足 | 增加原子层数 |
| 功函数异常 | 真空层不足 | 增加真空层厚度 |
| 偶极修正失效 | 偶极方向错误 | 检查IDIPOL设置 |
| 界面分离 | 初始距离太大 | 减小初始层间距 |
| 金属表面不收敛 | smearing不足 | 增加SIGMA或改变ISMEAR |

---

## 6. 参考文献

1. **Fiorentini, V., & Methfessel, M. (1996)**. Extracting convergent surface energies from slab calculations. *Journal of Physics: Condensed Matter*, 8(36), 6525.

2. **VASP Surface Calculations**: https://www.vasp.at/wiki/index.php/Surface_calculations

3. **QE Surface Tutorial**: https://www.quantum-espresso.org/Doc/surface.pdf

---

*文档版本: 1.0*
*更新日期: 2026-03-08*
