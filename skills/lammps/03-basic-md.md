# 03. 基础MD计算

## 目录
- [系综理论](#系综理论)
- [NVT系综](#nvt系综)
- [NPT系综](#npt系综)
- [积分器选择](#积分器选择)
- [邻居列表与截断](#邻居列表与截断)
- [输出与分析](#输出与分析)

---

## 系综理论

### 统计系综对照

| 系综 | 宏量固定 | 控制fix | 典型应用 |
|-----|---------|---------|---------|
| NVE | N(粒子), V(体积), E(能量) | nve | 微正则系综，验证守恒 |
| NVT | N, V, T(温度) | nvt / temp/* | 热力学性质计算 |
| NPT | N, P(压强), T | npt / press/* | 相变，密度计算 |
| NPH | N, P, H(焓) | nph | 冲击波模拟 |
| μVT | μ(化学势), V, T | gcmc | 吸附，扩散 |

---

## NVT系综

### 1. Nose-Hoover热浴

```lammps
# 标准Nose-Hoover (推荐)
fix 1 all nvt temp 300.0 300.0 100.0
#                   Tstart Tstop Tdamp

# Tdamp建议值: 100 * timestep
# 对于real单位制，timestep=1fs，Tdamp=100fs

# 温度斜坡
fix 1 all nvt temp 300.0 1000.0 100.0  # 从300K升温到1000K
```

### 2. Langevin动力学

```lammps
# Langevin热浴 (适合固体，不需要速度)
fix 1 all langevin 300.0 300.0 100.0 12345
#                       Tstart Tstop damp seed

fix 2 all nve          # 需要配合nve使用

# 部分原子热浴
fix 1 mobile langevin 300.0 300.0 100.0 12345
```

### 3. Berendsen热浴

```lammps
# Berendsen弱耦合 (快速平衡，不推荐用于采样)
fix 1 all temp/berendsen 300.0 300.0 100.0
fix 2 all nve
```

### 4. CSVR热浴

```lammps
# CSVR (随机速度重标定)
fix 1 all temp/csvr 300.0 300.0 100.0 12345
fix 2 all nve
```

### 5. 区域温控

```lammps
# 只控制特定区域温度
region hot block INF INF INF INF INF 10.0
fix 1 hot_atoms langevin 400.0 400.0 100.0 12345

# 层状系统不同温度
region 1 block INF INF INF INF 0 10
region 2 block INF INF INF INF 10 20
fix hot region_1 langevin 350 350 100 123
fix cold region_2 langevin 250 250 100 456
```

---

## NPT系综

### 1. Nose-Hoover恒压器

```lammps
# 各向同性压力控制
fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0
#                                          Pstart Pstop Pdamp

# 各向异性（独立控制xyz）
fix 1 all npt temp 300.0 300.0 100.0 aniso 1.0 1.0 1000.0

# 仅控制z方向（薄膜/界面）
fix 1 all npt temp 300.0 300.0 100.0 z 1.0 1.0 1000.0
```

### 2. Pdamp参数选择

```lammps
# Pdamp建议值: 1000 * timestep
# 过小的Pdamp导致压力振荡
# 过大的Pdamp导致响应缓慢

# 不同体系的典型值
# 液体/软材料: Pdamp = 500-1000 fs
# 固体/硬材料: Pdamp = 1000-5000 fs

# 耦合参数调整
fix 1 all npt temp 300.0 300.0 $(100.0*dt) iso 1.0 1.0 $(1000.0*dt)
```

### 3. 特殊压力控制

```lammps
# 三斜晶系（所有6个应力分量）
fix 1 all npt temp 300.0 300.0 100.0 tri 0.0 0.0 1000.0

# 固定形状，只控制体积
fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0 dilate partial

# 表面张力控制（薄膜）
fix 1 all npt temp 300.0 300.0 100.0 couple xy 0.0 0.0 1000.0
```

### 4. 膜/界面模拟

```lammps
# 表面张力计算 (γ = Lz * (Pzz - 0.5*(Pxx+Pyy)))
fix 1 all npt temp 300.0 300.0 100.0 x 1.0 1.0 1000.0 y 1.0 1.0 1000.0

# 固定xy面积，只放松z
fix 1 all npt temp 300.0 300.0 100.0 z 1.0 1.0 1000.0

# 使用npt/fdot控制表面张力
fix 1 all npt/fdot temp 300.0 300.0 100.0 fdot 0.0001
```

---

## 积分器选择

### 1. 标准积分器

```lammps
# 速度Verlet (默认，最稳定)
run_style verlet

# 可变速率Verlet (自动调整步长)
run_style respa 4 2 2 2 inner 2 4.0 6.0 middle 4 8.0 10.0 outer 12.0
```

### 2. 多时间步长RESPA

```lammps
# 分层积分策略
# Level 1: 键/角 (最频繁)
# Level 2: 非键近程
# Level 3: 长程库仑/远距离

run_style respa 3 2 2 inner 1 5.0 7.0 middle 2 9.0 outer 3

# 示例分解
# 内层: 2 fs (键/角/近程LJ)
# 中层: 4 fs (中程力)  
# 外层: 8 fs (长程kspace)
```

### 3. 约束动力学

```lammps
# SHAKE算法 (约束键/角)
fix 1 all shake 1.0e-4 100 0 b 1 2 a 1
#               tolerance max_iter output_every bond_types angle_types

# RATTLE (速度-Verlet版本)
fix 1 all rattle 1.0e-4 100 b 1 2

# Rigid约束 (刚体)
fix 1 water rigid/nvt molecule temp 300.0 300.0 100.0
fix 1 water rigid/npt molecule temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0

# 部分刚体
fix 1 solute rigid/nvt group 1 temp 300.0 300.0 100.0
```

### 4. 时间步长选择

```lammps
# 推荐时间步长 (real单位)
timestep 0.5    # 含氢体系，高能量
timestep 1.0    # 标准有机分子
timestep 2.0    #  coarse-grained
timestep 4.0    # 刚体水/约束体系

# metal单位
timestep 0.001  # 1 fs
timestep 0.002  # 2 fs

# 自适应步长
fix 1 all dt/reset 10 1.0e-5 0.001 0.01 emax 100.0
```

---

## 邻居列表与截断

### 1. 邻居列表设置

```lammps
# 标准设置
neighbor 2.0 bin          # 皮肤距离 2.0 Å
neigh_modify every 1 delay 0 check yes

# 大体系优化
neigh_modify every 1 delay 0 check yes page 100000 one 10000

# 减少重建频率（性能优化）
neigh_modify every 10 delay 0 check yes
```

### 2. 截断策略

```lammps
# 短程截断
pair_style lj/cut 10.0
pair_style buck/coul/cut 10.0 12.0

# 长程库仑
pair_style lj/cut/coul/long 10.0 12.0
kspace_style pppm 1.0e-4    # 精度控制

# PPPM参数优化
kspace_style pppm 1.0e-5
kspace_modify mesh 64 64 64 order 4
kspace_modify gewald 0.3

# 混合截断
pair_style hybrid lj/cut 10.0 morse 8.0 buck/coul/long 10.0 12.0
pair_coeff 1 1 lj/cut 0.2381 3.405
pair_coeff 2 2 morse 0.7102 1.6047 2.897
pair_coeff 1 2 buck/coul/long 10000.0 0.25 100.0
```

### 3. 长程库仑方法

```lammps
# PPPM (粒子-粒子粒子-网格) - 均匀体系
kspace_style pppm 1.0e-4
kspace_modify mesh 32 32 32 order 4

# PPPM/CG - 粗粒化体系
kspace_style pppm/cg 1.0e-4 1.0e-5

# Ewald求和
kspace_style ewald 1.0e-6

# MSM (多级求和) - 非周期体系
kspace_style msm 1.0e-4
kspace_modify boundary p p f  # 混合周期/非周期

# ScaFaCoS库
kspace_style scafacos fmm 1.0e-4
```

---

## 输出与分析

### 1. 热力学输出

```lammps
# 基本热力学量
thermo 100                    # 每100步输出
thermo_style custom step temp pe ke etotal press vol density

# 完整输出
thermo_style custom step atoms temp pe ke etotal ecoul elong press vol \
    lx ly lz xy xz yz cpu

# 多列格式
thermo_modify line multi format float %20.10g

# 自定义变量输出
variable my_press equal press
variable my_temp equal temp
thermo_style custom step v_my_press v_my_temp
```

### 2. 轨迹Dump

```lammps
# 标准dump
dump 1 all atom 100 dump.lammpstrj

# 自定义dump
dump 1 all custom 100 dump.custom id type x y z vx vy vz fx fy fz

# 图像dump
dump 2 all image 100 dump.*.jpg type type

# 电影dump
dump 3 all movie 100 movie.mp4 type type size 640 480

# NetCDF格式
dump 4 all netcdf 100 dump.nc id type x y z

# 压缩输出
dump 1 all custom 100 dump.gz id type x y z
```

### 3. 计算量定义

```lammps
# 温度
compute my_temp all temp

# 压强张量
compute my_press all pressure my_temp

# 径向分布函数
ccompute my_rdf all rdf 100 1 1 1 2 2 2
fix 1 all ave/time 100 5 1000 c_my_rdf[*] file rdf.dat mode vector

# 均方位移(MSD)
compute my_msd all msd
fix 2 all ave/time 10 100 1000 c_my_msd file msd.dat

# 空间分布
compute my_chunk all chunk/atom bin/1d x lower 5.0
fix 3 all ave/chunk 100 10 1000 my_chunk density/mass file density.dat

# 配位数
coord/atom cutoff 3.5
```

### 4. 分析Fix

```lammps
# 时间平均
fix 1 all ave/time 10 100 1000 c_my_temp file temp.dat

# 空间平均
fix 2 all ave/chunk 100 10 1000 my_chunk density/mass file density_profile.dat

# 历史记录
fix 3 all ave/histo 100 0.5 5.0 100 c_my_rdf file histo.dat

# 自相关函数
fix 4 all ave/correlate 10 100 1000 c_my_press[3] file vacf.dat type auto

# 动态矩阵
fix 5 all phonon 10 100 1000 map.in dynmat.dat
```

---

## 完整NVT/NPT示例

### NVT平衡示例

```lammps
# NVT平衡完整脚本
units metal
atom_style atomic
boundary p p p

read_data cu.lmp

pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# 能量最小化
minimize 1.0e-12 1.0e-12 1000 10000

# 初始速度
velocity all create 300.0 12345 mom yes rot yes dist gaussian

# NVT平衡
dump 1 all custom 100 nvt.dump id type x y z
dump_modify 1 sort id

thermo 100
thermo_style custom step temp pe ke etotal press

fix 1 all nvt temp 300.0 300.0 $(100.0*dt)

run 10000

unfix 1
undump 1

write_restart nvt.restart
```

### NPT平衡示例

```lammps
# NPT平衡完整脚本
units metal
atom_style atomic
boundary p p p

read_data cu.lmp

pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

# 长程力
kspace_style pppm 1.0e-4

# 温度控制 + 压力控制
velocity all create 300.0 12345

thermo 100
thermo_style custom step temp pe press vol density lx ly lz

dump 1 all custom 100 npt.dump id type x y z
dump_modify 1 sort id

# NPT各向同性
fix 1 all npt temp 300.0 300.0 $(100.0*dt) iso 0.0 0.0 $(1000.0*dt)

# 密度输出
variable dens equal density
fix 2 all ave/time 10 100 1000 v_dens file density.dat

run 50000

unfix 1
write_restart npt.restart
```

### 热循环示例

```lammps
# 升温-降温循环
variable a loop 5
label loop
    variable t equal 300+$a*100
    
    fix 1 all npt temp $t $t $(100.0*dt) iso 1.0 1.0 $(1000.0*dt)
    run 10000
    
    unfix 1
    write_restart equil_$a.restart
next a
jump SELF loop
```
