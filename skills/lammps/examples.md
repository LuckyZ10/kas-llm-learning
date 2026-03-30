# LAMMPS示例输入文件

> 包含各种模拟类型的示例输入脚本

---

## 1. Cu熔体模拟 (金属)

```lammps
# Cu熔体NVT模拟
# melt.in

units metal
atom_style atomic
boundary p p p

# 创建FCC晶格
lattice fcc 3.615
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 box

# Mishin Cu EAM势
pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

# 邻居列表
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# 能量最小化
minimize 1.0e-8 1.0e-8 1000 10000

# 升温到1800K熔化
velocity all create 1800.0 12345 mom yes rot yes dist gaussian

fix 1 all nvt temp 1800.0 1800.0 $(100.0*dt)

# 输出
dump 1 all custom 1000 melt.dump id type x y z vx vy vz
dump_modify 1 sort id

thermo 1000
thermo_style custom step temp pe ke etotal press vol

# 运行
timestep 0.001
run 50000

write_restart melt.restart
```

---

## 2. 水模拟 (SPC/E模型)

```lammps
# 水NPT模拟
# water.in

units real
atom_style full
boundary p p p

read_data spce.data

# SPC/E水模型
pair_style lj/cut/coul/long 10.0
pair_coeff 1 1 0.15535 3.166    # O
pair_coeff 2 2 0.0 0.0          # H

kspace_style pppm 1.0e-4

bond_style harmonic
bond_coeff 1 1000.0 1.0

angle_style harmonic
angle_coeff 1 1000.0 109.47

# 约束水
fix 1 all shake 1.0e-4 100 0 b 1 a 1

# 邻居列表
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# 能量最小化
minimize 1.0e-6 1.0e-8 1000 10000

# 平衡
velocity all create 300.0 12345 dist gaussian
fix 2 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0

# RDF计算
compute rdf_all all rdf 100 1 1 1 2 2 2
fix 3 all ave/time 100 10 10000 c_rdf_all[*] file rdf.dat mode vector

# 输出
dump 1 all custom 1000 water.dump id type x y z
dump_modify 1 sort id

thermo 1000
thermo_style custom step temp press pe density

timestep 1.0
run 100000
```

---

## 3. 伞形采样示例

```lammps
# 伞形采样 - 两个分子间距离
# umbrella.in

units real
atom_style full
boundary p p p

read_data two_molecules.data

# OPLS力场
pair_style lj/cut/coul/long 10.0
kspace_style pppm 1.0e-4

# 定义组
group mol1 id < 50
group mol2 id >= 50

# 初始分离距离
center_of_mass mol1
center_of_mass mol2

# 伞形势 - 通过fix spring/couple
fix umbrella mol1 spring/couple mol2 10.0 NULL NULL ${target_z} 0.0

# 或更精确的约束
fix umbrella mol1 spring/self 10.0

# 平衡
fix nvt all nvt temp 300.0 300.0 100.0

# 记录距离
variable d equal "sqrt((xcm(mol1,x)-xcm(mol2,x))^2 + (ycm(mol1,y)-xcm(mol2,y))^2 + (zcm(mol1,z)-xcm(mol2,z))^2)"
fix print_dist all print 100 "${d}" file distance_${target_z}.dat

timestep 1.0
run 100000
```

---

## 4. 副本交换MD

```lammps
# 温度副本交换
# remd.in

units real
atom_style full
boundary p p p

read_data peptide.data

# AMBER力场
include amber99sb.ff

# 温度列表
variable t world 300.0 340.0 385.0 435.0 490.0 550.0 615.0 685.0

# 初始化
velocity all create $t 12345 mom yes rot yes
fix 1 all nvt temp $t $t 100.0

# 副本交换
fix 2 all temper 1000 $t 12345 0 1

# 输出
dump 1 all custom 1000 remd_$t.dump id type x y z
thermo 1000

timestep 2.0
run 1000000
```

---

## 5. 金属拉伸变形

```lammps
# Cu单晶拉伸
# tensile.in

units metal
atom_style atomic
boundary p p p

read_data cu_crystal.data

# EAM势
pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

# 记录初始长度
variable Lx0 equal lx
variable strain equal "(lx-v_Lx0)/v_Lx0"
variable stress equal "-pxx/10000"  # GPa

# 能量最小化
minimize 1.0e-12 1.0e-12 1000 10000

# 初始速度
velocity all create 300.0 12345

# 平衡
fix 1 all npt temp 300.0 300.0 $(100.0*dt) y 0.0 0.0 $(1000.0*dt) z 0.0 0.0 $(1000.0*dt)
run 20000
unfix 1

# 应用拉伸
reset_timestep 0
fix 1 all deform 100 x erate 0.00001 units box
fix 2 all nvt temp 300.0 300.0 $(100.0*dt)

# 记录应力-应变
fix 3 all print 100 "${strain} ${stress}" file stress_strain.txt

# 输出
dump 1 all custom 1000 tensile.dump id type x y z
dump_modify 1 sort id

thermo 1000
thermo_style custom step temp v_strain v_stress pxx

timestep 0.001
run 100000
```

---

## 6. 石墨烯模拟

```lammps
# 石墨烯弛豫
# graphene.in

units metal
atom_style atomic
boundary p p p

# 创建石墨烯
lattice custom 2.46 a1 1.0 0.0 0.0 a2 0.5 0.866 0.0 a3 0.0 0.0 10.0 &
             basis 0.0 0.0 0.0 basis 0.333 0.667 0.0
region box block 0 40 0 40 0 1
create_box 1 box
create_atoms 1 box

# Tersoff势
pair_style tersoff
pair_coeff * * C.tersoff C

# 厚度调整 (2D)
change_box all z final -5 5

# 邻居列表
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# 能量最小化
minimize 1.0e-8 1.0e-8 1000 10000

# NPT平衡
velocity all create 300.0 12345
fix 1 all npt temp 300.0 300.0 $(100.0*dt) x 0.0 0.0 $(1000.0*dt) y 0.0 0.0 $(1000.0*dt)

# 输出
dump 1 all custom 1000 graphene.dump id type x y z
dump_modify 1 sort id

thermo 1000
thermo_style custom step temp pe press lx ly

timestep 0.001
run 100000
```

---

## 7. 并行运行脚本

```bash
#!/bin/bash
# run_parallel.sh

# MPI并行
mpirun -np 8 lmp -in input.lammps -log log.8

# MPI+OpenMP混合
export OMP_NUM_THREADS=4
mpirun -np 4 lmp -in input.lammps -sf omp -log log.4x4

# GPU加速
lmp -k on g 1 -sf kk -in input.lammps -log log.gpu

# 多GPU
mpirun -np 4 lmp -k on g 1 -sf kk -in input.lammps -log log.4gpu
```

---

## 8. 批量提交脚本 (PBS)

```bash
#!/bin/bash
#PBS -N lammps_job
#PBS -l nodes=2:ppn=16
#PBS -l walltime=24:00:00
#PBS -q normal

cd $PBS_O_WORKDIR

module load lammps/2024
module load openmpi/4.1

mpirun -np 32 lmp -in production.in -log production.log
```

---

## 9. 批量提交脚本 (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=lammps
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=normal

module load lammps/2024
module load openmpi/4.1

srun lmp -in production.in -log production.log
```

---

## 10. 数据文件生成 (Python)

```python
# generate_data.py
import numpy as np

def write_lammps_data(filename, positions, types, box, masses=None):
    """写入LAMMPS data文件"""
    n_atoms = len(positions)
    
    with open(filename, 'w') as f:
        f.write('LAMMPS data file\n\n')
        f.write(f'{n_atoms} atoms\n')
        f.write(f'{len(set(types))} atom types\n\n')
        
        # Box
        f.write(f'0.0 {box[0]} xlo xhi\n')
        f.write(f'0.0 {box[1]} ylo yhi\n')
        f.write(f'0.0 {box[2]} zlo zhi\n\n')
        
        # Masses
        if masses:
            f.write('Masses\n\n')
            for i, m in enumerate(masses, 1):
                f.write(f'{i} {m}\n')
            f.write('\n')
        
        # Atoms
        f.write('Atoms\n\n')
        for i, (pos, t) in enumerate(zip(positions, types), 1):
            f.write(f'{i} {t} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n')

# 生成FCC Cu
a = 3.615  # lattice constant
n = 5      # 5x5x5 unit cells

positions = []
types = []

for i in range(n):
    for j in range(n):
        for k in range(n):
            base = [i*a, j*a, k*a]
            # FCC basis
            for offset in [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]:
                pos = [base[m] + offset[m]*a for m in range(3)]
                positions.append(pos)
                types.append(1)

positions = np.array(positions)
box = [n*a, n*a, n*a]
masses = [63.546]  # Cu

write_lammps_data('cu_fcc.data', positions, types, box, masses)
print(f'Generated {len(positions)} atoms')
```
