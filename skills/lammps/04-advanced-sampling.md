# 04. 高级采样方法

## 目录
- [自由能计算概述](#自由能计算概述)
- [伞形采样 (Umbrella Sampling)](#伞形采样)
- [副本交换分子动力学 (REMD)](#副本交换分子动力学)
- [Metadynamics](#metadynamics)
- [Steered Molecular Dynamics](#steered-molecular-dynamics)
- [温度加速分子动力学 (TAD)](#温度加速分子动力学)

---

## 自由能计算概述

### 自由能与势 of Mean Force (PMF)

```
自由能计算框架:
├── 直接采样
│   └── 小体系，低能垒
├── 增强采样
│   ├── 伞形采样 (Umbrella Sampling)
│   ├── 副本交换 (REMD, REST)
│   ├── Metadynamics
│   └── 温度加速 (TAD, HTST)
└── 热力学积分
    ├── FEP (自由能微扰)
    └── TI (热力学积分)
```

### 集体变量 (Collective Variables)

```lammps
# Colvars包安装
make yes-colvars

# 基本CV定义 (colvars.txt)
colvar {
  name distance
  distance {
    group1 { atomNumbersRange 1-100 }
    group2 { atomNumbersRange 101-200 }
  }
}

colvar {
  name angle
  angle {
    group1 { atomNumbers 1 2 3 }
    group2 { atomNumbers 4 5 6 }
    group3 { atomNumbers 7 8 9 }
  }
}

colvar {
  name dihedral
  dihedral {
    group1 { atomNumbers 1 }
    group2 { atomNumbers 2 }
    group3 { atomNumbers 3 }
    group4 { atomNumbers 4 }
  }
}

colvar {
  name rmsd
  rmsd {
    atoms { atomNumbersRange 1-100 }
    refPositionsFile reference.pdb
  }
}

# LAMMPS中使用
fix 1 all colvars colvars.txt output output
```

---

## 伞形采样

### 1. 基本原理

```
添加偏置势能: U'(r) = U(r) + 0.5*k*(r-r0)^2

通过加权直方图分析(WHAM)或伞形积分重建PMF
```

### 2. LAMMPS实现

```lammps
# 方法1: 使用fix spring/self
fix 1 group1 spring/self 50.0        # k=50 kcal/mol/Å²

# 方法2: 使用fix colvars (推荐)
# colvars_umbrella.txt
colvar {
  name com_distance
  distance {
    group1 { atomNumbersRange 1-50 }
    group2 { atomNumbersRange 51-100 }
  }
}

harmonic {
  colvars com_distance
  centers 5.0        # r0 = 5 Å
  forceConstant 10.0 # k = 10 kcal/mol/Å²
}

# LAMMPS输入
fix 1 all colvars colvars_umbrella.txt output umbrella
```

### 3. 多窗口伞形采样

```bash
# 生成窗口脚本
for r0 in $(seq 2.0 0.5 10.0); do
    sed "s/CENTER/$r0/g" template.in > window_${r0}.in
done

# template.in内容:
# harmonic {
#   colvars com_distance
#   centers CENTER
#   forceConstant 10.0
# }
```

```lammps
# 单个窗口模拟脚本
units real
atom_style full
read_data system.data

pair_style lj/cut/coul/long 10.0
pair_coeff * * 0.1 3.0
kspace_style pppm 1.0e-4

timestep 1.0

# 设置伞形势
variable r0 equal $r          # 通过命令行传入
fix 1 all colvars colvars.in output umbrella_$r

# 平衡
thermo 100
run 10000

# 采样
run 100000

write_restart window_$r.restart
```

### 4. WHAM分析

```python
# wham_analysis.py
import numpy as np
from pymbar import MBAR

# 或独立WHAM程序
# https://membrane.urmc.rochester.edu/content/wham

# 数据格式: 每行一个窗口的CV值
# 2.1, 2.3, 2.0, 2.2, ...

# 运行WHAM
# wham [P_min] [P_max] [num_bins] [tol] [temperature] [metadatafile] [freefile]
# wham 2.0 10.0 100 0.0001 300 metadata.dat pmf.dat

# metadata.dat格式:
# window_2.0.dat 2.0 10.0
# window_2.5.dat 2.5 10.0
# ...
```

### 5. 完整伞形采样工作流

```bash
#!/bin/bash
# umbrella_workflow.sh

# 1. 准备窗口
R_MIN=2.0
R_MAX=10.0
N_WINDOWS=17
K=10.0

mkdir -p umbrella_windows
cd umbrella_windows

# 生成输入文件
i=0
for r in $(seq $R_MIN $((($R_MAX-$R_MIN)/$N_WINDOWS)) $R_MAX); do
    mkdir window_$i
    
    # 生成colvars文件
    cat > window_$i/colvars.in <<EOF
colvar {
  name d
  distance {
    group1 { atomNumbersRange 1-100 }
    group2 { atomNumbersRange 101-200 }
  }
}

harmonic {
  colvars d
  centers $r
  forceConstant $K
}
EOF

    # 生成LAMMPS输入
    cat > window_$i/run.in <<EOF
units real
atom_style atomic
read_data ../initial.data

pair_style lj/cut 10.0
pair_coeff * * 0.2381 3.405

fix 1 all colvars colvars.in output output

dump 1 all custom 100 dump.lammpstrj id type x y z

run 50000
EOF

    i=$((i+1))
done

# 2. 提交所有窗口
for d in window_*/; do
    (cd $d && mpirun -np 4 lmp -in run.in > log.out 2>&1 &)
done
wait

# 3. 运行WHAM
cd ..
wham $R_MIN $R_MAX 100 0.0001 300 umbrella_windows/metadata.dat pmf.dat
```

---

## 副本交换分子动力学

### 1. 温度副本交换 (T-REMD)

```lammps
# T-REMD输入文件模板
units real
atom_style full

read_data protein.data

# 力场设置
pair_style lj/cut/coul/long 10.0
pair_coeff * * 0.1 3.0
bond_style harmonic
angle_style harmonic
dihedral_style charmm
kspace_style pppm 1.0e-4

# 温度从命令行传入
variable t world 300.0 350.0 400.0 450.0 500.0 550.0 600.0

# 温度控制
velocity all create $t 12345 mom yes rot yes dist gaussian
fix 1 all nvt temp $t $t 100.0

# 输出
dump 1 all atom 1000 dump_$t.lammpstrj

# 副本交换设置
# 每1000步尝试交换
variable nswap equal 1000

# 使用fix temper
fix 2 all temper $nswap $t 12345 0 1 300.0 600.0

thermo 1000
run 1000000
```

### 2. 温度序列优化

```python
# 生成优化的温度序列
# 目标: 相邻温度间交换率20-30%

import numpy as np

def geometric_temperatures(T_min, T_max, n_replicas):
    """几何序列温度"""
    ratio = (T_max / T_min) ** (1.0 / (n_replicas - 1))
    temps = [T_min * (ratio ** i) for i in range(n_replicas)]
    return temps

def optimized_temperatures(T_min, T_max, n_replicas, alpha=0.05):
    """
    优化的温度序列
    考虑热容变化
    """
    temps = geometric_temperatures(T_min, T_max, n_replicas)
    # 进一步调整...
    return temps

# 示例
T_MIN = 300
T_MAX = 600
N_REP = 16

temps = geometric_temperatures(T_MIN, T_MAX, N_REP)
print(" ".join([f"{t:.1f}" for t in temps]))
```

### 3. 哈密顿副本交换 (H-REMD)

```lammps
# 使用不同力场参数的副本
# replica.in

variable rep world 0 1 2 3

if "${rep} == 0" then &\n    "pair_style lj/cut 10.0" &
    "pair_coeff * * 0.1 3.0"
  
if "${rep} == 1" then &\n    "pair_style lj/cut 10.0" &
    "pair_coeff * * 0.12 3.0"

# 继续定义其他副本...

# 交换设置
fix 1 all hremd 1000 $rep 4 0.1

run 100000
```

### 4. pH副本交换 (constant pH)

```lammps
# 需要特定包支持
# 或使用外部工具如CpHMD

# 基本思路: 交换质子化状态
fix 1 all phred 1000 7.0 12345 0 1
```

### 5. REST (副本交换溶质 tempering)

```lammps
# 只缩放溶质-溶质相互作用
# 更高效，需要更少副本

# 定义溶质和溶剂
group solute id < 100
group solvent id >= 100

# REST设置
variable lambda world 1.0 0.9 0.8 0.7 0.6

# 缩放势函数
pair_style lj/cut 10.0
pair_modify mix arithmetic scale $lambda

# 交换
fix 1 all rest 1000 $lambda 12345 0 1

run 100000
```

---

## Metadynamics

### 1. 使用PLUMED接口

```lammps
# 编译LAMMPS + PLUMED
make yes-plumed
make mpi

# LAMMPS输入
units real
atom_style full
read_data system.data

pair_style lj/cut/coul/long 10.0
kspace_style pppm 1.0e-4

fix 1 all nvt temp 300.0 300.0 100.0

# 启用PLUMED
fix 2 all plumed plumed.dat outfile plumed.out

run 1000000
```

```
# plumed.dat
# CV定义
dist: DISTANCE ATOMS=1,100

# Well-Tempered Metadynamics
metad: METAD ARG=dist SIGMA=0.2 HEIGHT=1.2 FILE=HILLS PACE=500 BIASFACTOR=10 TEMP=300

# 输出
PRINT ARG=dist,metad.bias STRIDE=100 FILE=COLVAR
```

### 2. 标准Metadynamics

```
# 标准Metadynamics (非well-tempered)
metad: METAD ARG=dist SIGMA=0.2 HEIGHT=1.2 FILE=HILLS PACE=500
```

### 3. 多CV Metadynamics

```
# plumed.dat
dist1: DISTANCE ATOMS=1,50
dist2: DISTANCE ATOMS=51,100
angle: ANGLE ATOMS=1,50,100

metad: METAD ARG=dist1,dist2 SIGMA=0.2,0.2 HEIGHT=1.2 FILE=HILLS PACE=500 BIASFACTOR=10

PRINT ARG=dist1,dist2,angle,metad.bias STRIDE=100 FILE=COLVAR
```

### 4. 重新加权与FES计算

```python
# 从HILLS文件计算自由能面
# sum_hills.py

import numpy as np
import matplotlib.pyplot as plt

def gaussian hill(x, center, sigma, height):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def compute_fes(hills_file, grid_min, grid_max, n_points=500):
    # 读取HILLS
    hills = np.loadtxt(hills_file, comments='#')
    
    # 创建网格
    grid = np.linspace(grid_min, grid_max, n_points)
    fes = np.zeros(n_points)
    
    # 累加高斯偏置
    for hill in hills:
        time, center, sigma, height, biasf = hill
        fes += gaussian(grid, center, sigma, height)
    
    # 取负号得到FES
    fes = -fes
    fes -= fes.min()  # 归一化
    
    return grid, fes

# 计算并绘图
grid, fes = compute_fes('HILLS', 0, 10)
plt.plot(grid, fes)
plt.xlabel('CV')
plt.ylabel('Free Energy (kJ/mol)')
plt.savefig('fes.png')
```

### 5. 收敛性分析

```python
# metad_convergence.py
# 分析偏置势收敛

import numpy as np
import matplotlib.pyplot as plt

def estimate_error(hills_file, time_windows):
    """估计不同时间窗口的FES误差"""
    errors = []
    
    for t_max in time_windows:
        # 读取到t_max的HILLS
        hills = np.loadtxt(hills_file, comments='#')
        hills_subset = hills[hills[:, 0] <= t_max]
        
        # 计算FES
        # ...
        
        errors.append(error)
    
    return time_windows, errors

# 绘制收敛曲线
plt.plot(time_windows, errors)
plt.xlabel('Simulation Time')
plt.ylabel('FES Error')
```

---

## Steered Molecular Dynamics

### 1. 恒力牵引

```lammps
# 使用fix spring/self进行SMD
# 或fix colvars

# colvars_smd.txt
colvar {
  name dist
  distance {
    group1 { atomNumbersRange 1-10 }
    group2 { atomNumbersRange 100-110 }
  }
}

harmonicWalls {
  colvars dist
  lowerWalls 2.0
  upperWalls 10.0
  lowerWallConstant 10.0
  upperWallConstant 10.0
}

# 或恒速度牵引
# smd.txt
colvar {
  name com_dist
  distance {
    group1 { atomNumbersRange 1-100 }
    group2 { atomNumbersRange 101-200 }
  }
}

smd {
  colvars com_dist
  outputCenters on
  outputAccumulatedWork on
  centers 5.0
  targetCenters 15.0
  targetNumSteps 50000
  forceConstant 10.0
}
```

### 2. 恒速度SMD

```lammps
# 直接实现恒速度SMD
variable t equal step
variable x0 equal 5.0
variable v equal 0.001       # Å/fs
variable target equal v_tstep*$v + x0

fix 1 group1 spring couple group2 10.0 ${target} 0.0 0.0 0

# 输出功
variable smd_force equal fcm(group1,x)
variable smd_work equal v_smd_force*(xcm(group1,x)-xcm(group1,x))
fix 2 all print 100 "${t} ${target} $(xcm(group1,x)) ${smd_force} ${smd_work}" file smd.dat
```

### 3. Jarzynski等式分析

```python
# jarzynski_analysis.py
import numpy as np

# 读取多个SMD轨迹的功
def read_work_values(file_pattern, n_runs):
    works = []
    for i in range(n_runs):
        data = np.loadtxt(file_pattern % i)
        # 最后一列是累积功
        works.append(data[-1, -1])
    return np.array(works)

# Jarzynski等式
def jarzynski_free_energy(works, T=300.0):
    kB = 0.001987  # kcal/mol/K
    beta = 1.0 / (kB * T)
    
    # ΔG = -kT ln⟨exp(-βW)⟩
    free_energy = -np.log(np.mean(np.exp(-beta * works))) / beta
    return free_energy

# 累积量展开 (更稳定的估计)
def cumulant_expansion(works, T=300.0):
    kB = 0.001987
    beta = 1.0 / (kB * T)
    
    W_mean = np.mean(works)
    W_var = np.var(works)
    
    # 二阶累积量
    delta_G = W_mean - 0.5 * beta * W_var
    return delta_G

# 使用示例
works = read_work_values("smd_run_%d.dat", 20)
dG_jarzynski = jarzynski_free_energy(works)
dG_cumulant = cumulant_expansion(works)

print(f"ΔG (Jarzynski): {dG_jarzynski:.2f} kcal/mol")
print(f"ΔG (Cumulant): {dG_cumulant:.2f} kcal/mol")
```

---

## 温度加速分子动力学

### 1. Hyperdynamics

```lammps
# 使用提升势加速稀有事件
# 需要boost potential

# 局部超动力学
fix 1 all hyper/global 0.8 12345  # V_max = 0.8 eV

# 或局部超动力学
fix 1 all hyper/local 0.8 12345 0.1  # V_max, seed, sigma

run 1000000
```

### 2. 并行复本动力学 (PRD)

```lammps
# PRD设置
# prd.in

units metal
atom_style atomic

read_data initial.data

pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

# PRD参数
variable nrep equal 4          # 副本数
variable t_corr equal 1000     # 关联时间
variable t_dephase equal 100   # 去相位时间

# 定义事件检测
compute pe all pe/atom
dump 1 all custom 100 event.dump id type x y z c_pe

fix 1 all prd ${nrep} ${t_corr} ${t_dephase} event.dump NULL dephase temp

run 1000000
```

### 3. 温度加速分子动力学 (TAD)

```lammps
# TAD输入
units metal
atom_style atomic

read_data initial.data

pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

# TAD参数
variable t_high equal 800.0    # 高温 (K)
variable t_low equal 300.0     # 目标温度 (K)
variable delta equal 0.005     # 置信参数

# 最小化
minimize 1.0e-8 1.0e-8 1000 10000

# TAD运行
fix 1 all tad ${t_low} ${t_high} ${delta} 12345

run 10000000
```

### 4. 加速时间估计

```python
# 计算加速因子
# acceleration_factor.py

import numpy as np

kB = 8.617e-5  # eV/K

def hyperdynamics_acceleration(V_boost, T):
    """计算超动力学加速因子"""
    return np.exp(V_boost / (kB * T))

def tad_acceleration(T_high, T_low, barrier):
    """计算TAD加速因子"""
    # 高温加速
    high_temp_boost = np.exp(-barrier / (kB * T_high))
    low_temp_rate = np.exp(-barrier / (kB * T_low))
    return low_temp_rate / high_temp_boost

# 示例
V_boost = 0.5  # eV
T = 300  # K
accel = hyperdynamics_acceleration(V_boost, T)
print(f"Hyperdynamics acceleration: {accel:.2e}x")

# TAD示例
T_high = 800
T_low = 300
barrier = 1.0  # eV
accel_tad = tad_acceleration(T_high, T_low, barrier)
print(f"TAD acceleration: {accel_tad:.2e}x")
```

---

## 完整增强采样工作流

### 1. 选择合适的增强采样方法

| 问题类型 | 推荐方法 | 说明 |
|---------|---------|------|
| 已知反应坐标 | 伞形采样 | 精确PMF |
| 复杂能量面 | REMD | 温度跳跃 |
| 未知反应坐标 | Metadynamics | 自动探索 |
| 强制解离/折叠 | SMD | 快速采样 |
| 稀有事件 | TAD/Hyperdynamics | 时间加速 |
| 柔性分子 | REST | 高效采样 |

### 2. 增强采样最佳实践

```lammps
# 组合方法示例
# 先用Metadynamics探索，再用伞形采样细化

# 阶段1: Metadynamics快速探索
# plumed_explore.dat
metad: METAD ARG=cv1,cv2 SIGMA=0.5,0.5 HEIGHT=2.0 PACE=100 FILE=HILLS_TEMP

# 阶段2: 基于HILLS确定过渡态
# 分析HILLS找到鞍点

# 阶段3: 伞形采样细化
# colvars_refine.txt
harmonic {
  colvars cv1
  centers [TS位置]
  forceConstant 50.0  # 更强的约束
}
```
