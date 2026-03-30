# 06. 材料体系案例

> 涵盖金属、聚合物、生物分子、界面体系的LAMMPS模拟实战

---

## 目录
- [金属体系](#金属体系)
- [聚合物体系](#聚合物体系)
- [生物分子体系](#生物分子体系)
- [界面体系](#界面体系)
- [复合材料](#复合材料)

---

## 金属体系

### 1. 单晶Cu的力学性能

```lammps
# Cu单晶拉伸模拟
units metal
atom_style atomic
boundary p p p

# 创建FCC晶格
lattice fcc 3.615
region box block 0 20 0 20 0 40
create_box 1 box
create_atoms 1 box

# Mishin Cu EAM势
pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# 能量最小化
minimize 1.0e-12 1.0e-12 1000 10000

# 初始温度300K
velocity all create 300.0 12345 mom yes rot yes

# NPT平衡
dump 1 all custom 1000 equil.dump id type x y z
dump_modify 1 sort id

fix 1 all npt temp 300.0 300.0 $(100.0*dt) iso 0.0 0.0 $(1000.0*dt)
thermo 1000
thermo_style custom step temp pe press vol lx ly lz

run 20000
unfix 1
undump 1

# 应用拉伸变形 - x方向
reset_timestep 0

# 计算工程应变
variable strain equal "(lx - v_Lx0)/v_Lx0"
variable stress equal "-pxx/10000"  # 转换为GPa

# 记录初始长度
variable Lx0 equal lx

# 施加变形 - 使用deform
fix 1 all deform 100 x erate 0.00001 units box  # 10^7 s^-1 应变速率
fix 2 all nvt temp 300.0 300.0 $(100.0*dt)

# 输出应力-应变
fix 3 all print 100 "${strain} ${stress}" file stress_strain.txt screen no

dump 2 all custom 1000 deform.dump id type x y z
dump_modify 2 sort id

thermo 1000
thermo_style custom step temp v_strain v_stress pxx press

run 100000

write_restart deform.restart
```

**应力-应变曲线分析:**

```python
# analyze_stress_strain.py
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = np.loadtxt('stress_strain.txt')
strain = data[:, 0]
stress = data[:, 1]

# 计算弹性模量 (线性段)
linear_region = strain < 0.02
E = np.polyfit(strain[linear_region], stress[linear_region], 1)[0]
print(f"Young's Modulus: {E:.2f} GPa")

# 屈服强度 (0.2%偏移法)
offset_strain = strain - 0.002
yield_idx = np.where(stress < E * offset_strain)[0][0]
yield_strength = stress[yield_idx]
print(f"Yield Strength: {yield_strength:.2f} GPa")

# 极限抗拉强度
uts = np.max(stress)
print(f"UTS: {uts:.2f} GPa")

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(strain, stress, 'b-', linewidth=2)
plt.axhline(y=yield_strength, color='r', linestyle='--', label=f'Yield: {yield_strength:.2f} GPa')
plt.axhline(y=uts, color='g', linestyle='--', label=f'UTS: {uts:.2f} GPa')
plt.xlabel('Strain')
plt.ylabel('Stress (GPa)')
plt.title('Cu Single Crystal Stress-Strain Curve')
plt.legend()
plt.grid(True)
plt.savefig('stress_strain.png', dpi=300)
```

### 2. Ni-Al合金的沉淀强化

```lammps
# Ni-Al合金模拟
units metal
atom_style atomic
boundary p p p

# 创建Ni基体含Al沉淀
lattice fcc 3.52
region box block 0 30 0 30 0 30

create_box 2 box
create_atoms 1 box

# 随机替换部分Ni为Al (Ni3Al成分)
set type 1 type/subset 2 25 239823  # 25% Al

# 合金EAM势 (Mishin)
pair_style eam/alloy
pair_coeff * * NiAl.eam.alloy Ni Al

# 退火处理
velocity all create 1500.0 12345
fix 1 all npt temp 1500.0 1500.0 0.1 iso 0.0 0.0 1.0
dump 1 all custom 1000 anneal.dump id type x y z

run 50000  # 50 ps退火

# 淬火
unfix 1
fix 1 all npt temp 1500.0 300.0 0.1 iso 0.0 0.0 1.0
run 200000

# 分析沉淀相
compute pe_atom all pe/atom
dump 2 all custom 1000 precipitate.dump id type x y z c_pe_atom
```

### 3. 位错滑移模拟

```lammps
# Cu中刃位错滑移
units metal
atom_style atomic

# 读取预定位错结构
read_data dislocation.data

# EAM势
pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

# 应用剪切应力
variable applied_shear equal 0.001  # 逐步增加

fix 1 all addforce v_applied_shear 0.0 0.0
fix 2 all nvt temp 300.0 300.0 0.1

# 追踪位错位置
dump 1 all custom 100 disloc.dump id type x y z
dump_modify 1 sort id

# 计算位错速度
compute disloc_pos all reduce x y z
fix 3 all ave/time 10 100 1000 c_disloc_pos file disloc_position.dat

thermo 1000
run 100000
```

### 4. 晶界迁移

```lammps
# 双晶Cu - Sigma5(310)晶界
units metal
atom_style atomic

read_data bicrystal.data

pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

# 设置温度梯度驱动晶界迁移
region hot block INF INF INF INF 0 10
region cold block INF INF INF INF 40 50

fix 1 hot_atoms langevin 600 600 0.1 12345
fix 2 cold_atoms langevin 300 300 0.1 23456
fix 3 all nve

# 追踪晶界位置
dump 1 all custom 1000 gb.dump id type x y z
dump_modify 1 sort id

# 计算晶界速度
run 200000
```

---

## 聚合物体系

### 1. 聚乙烯(PE)熔体模拟

```lammps
# PE熔体 - 粗粒化模型
units real
atom_style full
boundary p p p

# 读取粗粒化PE链
read_data pe_melt.data

# OPLS-AA力场
pair_style lj/cut/coul/long 10.0
pair_coeff * * 0.07 3.55  # CH2 united atom
kspace_style pppm 1.0e-4

# 键合项
bond_style harmonic
bond_coeff 1 350.0 1.53  # C-C bond

angle_style harmonic
angle_coeff 1 60.0 109.5  # C-C-C angle

dihedral_style opls
dihedral_coeff 1 1.74 -0.157 0.279 0.00  # C-C-C-C

# 温度循环
velocity all create 600.0 12345
fix 1 all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0

# 输出密度
dump 1 all custom 1000 pe.dump id type x y z
fix 2 all ave/time 100 10 1000 density file density.dat

thermo 1000
run 200000  # 平衡

# 降温到300K
unfix 1
fix 1 all npt temp 600.0 300.0 100.0 iso 1.0 1.0 1000.0
run 500000  # 结晶过程
```

### 2. 聚合物玻璃化转变

```lammps
# PS玻璃化转变
units real
atom_style full

read_data ps_system.data

# GAFF力场
pair_style lj/cut/coul/long 10.0
kspace_style pppm 1.0e-4

bond_style harmonic
angle_style harmonic
dihedral_style charmm
improper_style cvff

# 降温循环
variable T equal 500
dump 1 all custom 1000 ps_T$T.dump id type x y z

label temp_loop
fix 1 all npt temp $T $T 100.0 iso 1.0 1.0 1000.0
run 50000

unfix 1
variable T equal $T-25
if "$T > 250" then "jump SELF temp_loop"

# 分析比容vs温度
```

### 3. 聚合物纳米复合材料

```lammps
# SiO2纳米颗粒/PE基体
units real
atom_style full

read_data nanocomposite.data

# 多力场混合
pair_style hybrid lj/cut 10.0 buck/coul/long 10.0 12.0
pair_coeff 1 1 lj/cut 0.07 3.55  # PE-PE
pair_coeff 2 2 buck/coul/long 10000.0 0.22 32.0  # SiO2-SiO2
pair_coeff 1 2 lj/cut 0.1 3.5  # PE-SiO2

# 计算界面能
compute pe_atom all pe/atom
dump 1 all custom 1000 nano.dump id type x y z c_pe_atom

# 应变分析
fix 1 all npt temp 300.0 300.0 100.0 aniso 1.0 1.0 1000.0
run 100000
```

---

## 生物分子体系

### 1. 蛋白质折叠 (Chignolin)

```lammps
# Chignolin小蛋白折叠
units real
atom_style full

# 读取PDB
read_data chignolin.data

# AMBER99SB-ILDN力场
include ff.AMBER99SB

# GBSA隐式溶剂
fix 1 all gb/sa

# 副本交换设置
variable r world 300.0 330.0 365.0 400.0 440.0 485.0 535.0 590.0 650.0

velocity all create $r 12345 mom yes
fix 2 all nvt temp $r $r 100.0

# 温度交换
fix 3 all temper 1000 $r 12345

# 输出RMSD
dump 1 all custom 1000 fold.dump id type x y z
dump_modify 1 sort id

thermo 1000
run 10000000
```

### 2. 脂质双层膜

```lammps
# POPC脂质双层
units real
atom_style full

read_data membrane.data

# CHARMM36力场
include ff.CHARMm36

# 限制水分子
group water type 80 81  # TIP3P
fix 1 water shake 1.0e-4 100 0 b 1 a 1

# NPT各向异性
fix 2 all npt temp 310.0 310.0 100.0 \
    x 1.0 1.0 1000.0 y 1.0 1.0 1000.0 z 1.0 1.0 1000.0

# 计算膜厚度
dump 1 all custom 1000 membrane.dump id type x y z

# 面积每脂质
compute lipid_area all reduce ave lx*ly
dump 2 all custom 1000 area.dat c_lipid_area

thermo 1000
run 2000000
```

### 3. DNA-蛋白质相互作用

```lammps
# DNA-转录因子复合物
units real
atom_style full

read_data dna_protein.data

# AMBER力场
include parmbsc1.dat

# 约束DNA骨架
group dna_backbone type 1 2 3
group protein id > 1000

fix 1 dna_backbone spring/self 10.0

# 蛋白接近DNA
fix 2 protein smd cvel 100.0 0.0001 couple dna group 1 1.0

thermo 1000
run 500000
```

---

## 界面体系

### 1. 金属-水界面

```lammps
# Cu(111)-水界面
units real
atom_style full

read_data cu_water_interface.data

group cu type 1
group water type 2 3

# 金属EAM
pair_style hybrid eam/alloy lj/cut/coul/long 10.0
pair_coeff 1 1 eam/alloy Cu_u3.eam.alloy Cu
pair_coeff 2 2 lj/cut/coul/long 0.15535 3.166  # O
pair_coeff 3 3 lj/cut/coul/long 0.0 0.0        # H
pair_coeff 1 2 lj/cut/coul/long 0.5 3.0        # Cu-O
kspace_style pppm 1.0e-4

# 刚性水
fix 1 water shake 1.0e-4 100 0 b 1 a 1

# 固定金属底层
region fixed block INF INF INF INF INF 5.0
group fixed region fixed
fix 2 fixed setforce 0.0 0.0 0.0

# 界面水分析
compute water_pe water pe/atom
dump 1 all custom 1000 interface.dump id type x y z c_water_pe

thermo 1000
run 500000
```

### 2. 液-液界面 (油-水)

```lammps
# 癸烷-水界面
units real
atom_style full

read_data decane_water.data

# OPLS力场
pair_style lj/cut/coul/long 10.0
kspace_style pppm 1.0e-4

# 界面张力计算
fix 1 all npt temp 300.0 300.0 100.0 \
    x 1.0 1.0 1000.0 y 1.0 1.0 1000.0

# 压力张量
compute pressure all pressure thermo_temp
fix 2 all ave/time 100 10 1000 c_pressure file pressure.dat

thermo 1000
run 500000
```

### 3. 固-液-气三相接触线

```lammps
# 水滴在固体表面
units real
atom_style full

read_data droplet_on_surface.data

group substrate type 1
group droplet type 2 3

# 约束接触角
fix 1 droplet wall/region lower zlo 2.0 1.0 1.0 0.5

# 测量接触角
dump 1 all custom 1000 contact.dump id type x y z

thermo 1000
run 1000000
```

---

## 复合材料

### 1. CNT/聚合物复合材料

```lammps
# CNT增强环氧树脂
units real
atom_style full

read_data cnt_composite.data

group cnt type 1  # 碳纳米管
group polymer type 2-10

# AIREBO + OPLS
pair_style hybrid airebo 3.0 lj/cut 10.0
pair_coeff 1 1 airebo CH.airebo C
pair_coeff * * lj/cut 0.1 3.5

# 界面剪切强度
fix 1 cnt move linear 0.001 0.0 0.0
fix 2 polymer nvt temp 300.0 300.0 100.0

# 计算拉力
variable fcnt equal fcm(cnt,x)
fix 3 all print 100 "$(step) ${fcnt}" file pull_force.dat

thermo 1000
run 500000
```

### 2. 石墨烯/金属界面

```lammps
# 石墨烯/Cu界面
units metal
atom_style atomic

read_data graphene_cu.data

group graphene id <= 200
group copper id > 200

# 混合势
pair_style hybrid eam/alloy tersoff lj/cut 10.0
pair_coeff * * eam/alloy Cu_u3.eam.alloy NULL Cu
pair_coeff 1 1 tersoff C.tersoff C
pair_coeff 1 2 lj/cut 0.02 3.0

# 界面结合能
compute pe_graphene graphene pe/atom
compute pe_copper copper pe/atom

dump 1 all custom 1000 interface.dump id type x y z

thermo 1000
run 100000
```

### 3. 多尺度纤维增强

```lammps
# 纤维/基体/界面多尺度
units real
atom_style full

read_data fiber_composite.data

# 不同区域不同力场
# 纤维: 全原子
# 基体: 粗粒化
# 界面: 混合

# 使用atom_style hybrid

# 应力传递分析
fix 1 all npt temp 300.0 300.0 100.0 x 1.0 1.0 1000.0

dump 1 all custom 1000 composite.dump id type x y z

thermo 1000
run 500000
```

---

## 材料案例快速参考

| 材料类型 | 推荐力场 | 系综 | 关键分析 |
|---------|---------|------|---------|
| 金属(Cu,Al) | EAM/Mishin | NPT | 应力-应变, 缺陷演化 |
| 合金(Ni-Al) | MEAM | NPT | 沉淀相, 相图 |
| 聚合物(PE,PS) | OPLS/GAFF | NPT | Tg, 结晶度 |
| 生物分子 | AMBER/CHARMM | NVT/NPT | RMSD, Rg |
| 界面 | 混合势 | NVT | 界面能, 接触角 |
| 复合材料 | hybrid | NPT | 应力传递, 界面强度 |

---

## 分析脚本汇总

### 常见材料分析

```python
# materials_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry.analysis import Analysis

def analyze_metal_deformation(dump_files):
    """分析金属变形"""
    # 位错密度, 晶粒取向
    pass

def analyze_polymer_crystallization(dump_files):
    """分析聚合物结晶"""
    # 取向序参数, 密度
    pass

def analyze_interface_energy(dump_files):
    """计算界面能"""
    # 分离功, 界面张力
    pass

def analyze_stress_transfer(dump_files):
    """分析应力传递"""
    # 局部应力分布
    pass
```
