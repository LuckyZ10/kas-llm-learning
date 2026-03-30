# 02. 势函数库 (Potentials)

## 目录
- [势函数概述](#势函数概述)
- [经典力场](#经典力场)
- [机器学习势](#机器学习势)
- [ReaxFF反应力场](#reaxff反应力场)
- [EAM/MEAM势](#eammbeam势)
- [势函数选择与验证](#势函数选择与验证)

---

## 势函数概述

LAMMPS支持多种描述原子间相互作用的势函数形式：

```
势函数类型分布:
├── 经典力场 (Classical)
│   ├── 两体势: LJ, Morse, Buckingham
│   ├── 多体势: EAM, MEAM, Tersoff, SW
│   └── 分子力场: CHARMM, AMBER, OPLS, COMPASS
├── 机器学习势 (ML)
│   ├── ACE (ML-PACE)
│   ├── POD (ML-POD)
│   ├── SNAP
│   ├── DeepMD (外部)
│   └── NequIP/MACE (通过接口)
└── 反应力场 (Reactive)
    ├── ReaxFF
    ├── COMB
    └── BOP
```

---

## 经典力场

### 1. Lennard-Jones (LJ) 势

```lammps
# 基本LJ势
pair_style lj/cut 10.0
pair_coeff 1 1 0.2381 3.405   # ε σ (Ar)

# 截断版本
pair_style lj/cut/coul/long 10.0 12.0   # LJ截断 库仑长程
pair_style lj/smooth 8.0 10.0           # 平滑截断
pair_style lj/expand 10.0               # 可变粒子尺寸
```

| 体系 | ε (kcal/mol) | σ (Å) | 来源 |
|-----|-------------|-------|------|
| Ar | 0.238 | 3.405 | [Maitland et al] |
| Ne | 0.069 | 2.749 | [Maitland et al] |
| Kr | 0.449 | 3.60 | [Maitland et al] |
| CH₄ | 0.294 | 3.81 | TraPPE |

### 2. Buckingham势

```lammps
# A*exp(-r/ρ) - C/r^6
pair_style buck/coul/long 10.0 12.0
pair_coeff 1 1 10000.0 0.25 100.0   # A ρ C

# 常用氧化物参数 (Catti et al.)
pair_coeff * * buck 9547.96 0.2192 32.0   # O-O
pair_coeff 1 2 buck 2088.8 0.2649 0.0     # Si-O
```

### 3. Morse势

```lammps
# D*[1 - exp(-α(r-r₀))]^2
pair_style morse 10.0
pair_coeff 1 1 0.7102 1.6047 2.897   # D α r₀ (H-H bond)
```

### 4. Stillinger-Weber (SW) 势

```lammps
# 适用于硅等共价晶体
pair_style sw
pair_coeff * * Si.sw Si

# Si.sw文件内容
Si Si Si 7.049556277 0.6022245584 ...
```

### 5. Tersoff势

```lammps
# 适用于碳、硅、锗等
pair_style tersoff
pair_coeff * * SiC.tersoff Si C

# 多元素体系
pair_coeff * * SiCGe.tersoff Si C Ge
```

### 6. 分子力场 - CHARMM/AMBER/OPLS

```lammps
# CHARMM力场
pair_style lj/charmm/coul/charmm 8.0 10.0
bond_style harmonic
angle_style harmonic
dihedral_style charmm
improper_style harmonic

# AMBER力场
pair_style lj/cut/coul/long 10.0
bond_style harmonic
angle_style harmonic
dihedral_style charmm
improper_style cvff

# OPLS-AA
pair_style lj/cut/coul/long 10.0
bond_style harmonic
angle_style harmonic
dihedral_style opls
```

### 7. 水模型

```lammps
# SPC/E水模型
pair_style lj/cut/coul/long 10.0
pair_coeff 1 1 0.15535 3.166    # O
pair_coeff 2 2 0.0 0.0          # H (无LJ)
bond_style harmonic
bond_coeff 1 1000.0 1.0         # O-H
angle_style harmonic
angle_coeff 1 1000.0 109.47     # H-O-H

# TIP3P/TIP4P/TIP5P
pair_style lj/cut/coul/long/tip4p/long 10.0   # TIP4P
fix 1 all shake 1.0e-4 100 0 b 1 a 1          # 约束水

# 刚性水模型
fix 1 all rigid/nvt molecule temp 300.0 300.0 100.0
```

---

## 机器学习势

### 1. ACE (Atomic Cluster Expansion)

```lammps
# ML-PACE包
pair_style pace
pair_coeff * * Cu.yaml Cu

# 高阶ACE
pair_style pace/extrapolation 10.0  # 含外推警告
pair_coeff * * Ni_ace.yace Ni

# YAML格式势文件示例
# B_species: [Cu]
# deltaSplineBins: 0.001
# elements: [Cu]
# embeddings:
#   ndensity: 1
#   rho_core_cut: 200000
# ...
```

### 2. POD (Proper Orthogonal Descriptors)

```lammps
# ML-POD包
pair_style pod
pair_coeff * * Ta.pod Ta_coefficients.pod Ta
```

### 3. SNAP (Spectral Neighbor Analysis Potential)

```lammps
# SNAP势
pair_style snap
pair_coeff * * Ta06A.snapcoeff Ta06A.snapparam Ta

# snapparam文件
# rcutfac 4.67637
# twojmax 6
# rfacc0 0.99363
# rmin0 0.0
# switchflag 1
# bzeroflag 0
# quadraticflag 0
```

### 4. DeepMD-kit接口

```lammps
# DeepMD包 (需单独编译)
pair_style deepmd graph.pb
pair_coeff * * 

# 多模型集成
pair_style deepmd graph1.pb graph2.pb graph3.pb out_freq 10 out_file md.out
```

### 5. NequIP/MACE (通过外部接口)

```python
# Python接口使用NequIP
from lammps import lammps
import torch
from nequip.model import model_from_config

# 加载NequIP模型
model = model_from_config(config)

# 在LAMMPS中使用自定义力计算
lmp = lammps()
# ... 通过fix python/invoke 调用
```

### ML势性能对比

| 势类型 | 精度 | 速度 | 训练数据量 | 适用体系 |
|-------|------|-----|-----------|---------|
| SNAP | 高 | 快 | 中(~100) | 单元素/合金 |
| ACE | 极高 | 中 | 中(~100) | 多元体系 |
| POD | 极高 | 中 | 中(~100) | 多元体系 |
| DeepMD | 极高 | 慢 | 大(~1000) | 任意体系 |
| NequIP | 极高 | 慢 | 中(~100) | 复杂化学 |
| MACE | 极高 | 中 | 中(~100) | 通用 |

---

## ReaxFF反应力场

### 1. 基本使用

```lammps
# 启用ReaxFF
pair_style reax/c NULL
pair_coeff * * ffield.reax.CHONSSi C H O N

# 必需: 能量最小化和平滑
fix 1 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c
fix 2 all reax/c/species 10 10 100 species.tatb

# 温度控制（使用较低阻尼）
fix 3 all temp/berendsen 300.0 300.0 100.0
```

### 2. 常见ReaxFF力场文件

| 力场文件 | 元素覆盖 | 应用 |
|---------|---------|------|
| ffield.reax.CHONSSi | C,H,O,N,S,Si | 有机物、含能材料 |
| ffield.reax.FeOCH | Fe,O,C,H | 铁氧化、腐蚀 |
| ffield.reax.CuCH | Cu,C,H | 铜催化 |
| ffield.reax.NiCHO | Ni,C,H,O | 镍催化 |
| ffield.reax.AlNiO | Al,Ni,O | 铝热反应 |

### 3. ReaxFF参数优化

```lammps
# 控制化学精度
echo both
units real
atom_style charge

read_data data.reax

# 电荷平衡收敛
fix 1 all qeq/reax 1 0.0 10.0 1.0e-6 param.qeq

# 轨迹输出含键信息
compute reax all reax/c/bonds
fix 5 all reax/c/bonds 100 bonds.reax

# 物种识别
fix 6 all reax/c/species 100 100 500 species.out element C H O N
```

---

## EAM/MEAM势

### 1. EAM (Embedded Atom Method)

```lammps
# 基础EAM
pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

# 多元素EAM
pair_style eam/alloy
pair_coeff * * CuAg.eam.alloy Cu Ag

# 常用EAM库
# - Mishin: Ni, Al, Cu, 合金
# - Zhou: 大量单质和合金
# - Mendelev: Fe, 玻璃形成体系
```

### 2. MEAM (Modified EAM)

```lammps
# MEAM
pair_style meam
pair_coeff * * library.meam Ni Ni.meam Ni

# 第二近邻MEAM (2NN MEAM)
pair_style meam/spline
pair_coeff * * Ti.meam.spline Ti

# 多元素MEAM
pair_style meam
pair_coeff * * library.meam Al Si C AlSiC.meam Al Si C
```

### 3. 常见EAM数据库

```bash
# NIST Interatomic Potentials
wget https://www.ctcms.nist.gov/potentials/Download/Al-Mg/Al-Mg-2009--Mendelev-M-I--Al-Mg/setfl/Al-Mg-2009--Mendelev-M-I--Al-Mg--ipr1/MgAl09.eam.alloy

# OpenKIM数据库
# https://openkim.org/browse/models
pair_style kim Al_EAM_Dynamo_ErcolessiAdams_1994__MO_123629422043_005
pair_coeff * * Al
```

---

## 势函数选择与验证

### 1. 选择流程图

```
体系类型判断:
├── 纯金属/合金
│   ├── 简单结构 → EAM/MEAM
│   └── 含化学反应 → ReaxFF
├── 共价晶体 (Si, C)
│   ├── 完美晶体 → Tersoff/SW
│   └── 缺陷/相变 → ML势
├── 离子晶体
│   └── Buckingham/组合势
├── 分子/有机体系
│   ├── 小分子 → OPLS/GAFF
│   ├── 生物分子 → CHARMM/AMBER
│   └── 含反应 → ReaxFF
└── 未知/复杂体系
    └── ML势 (ACE/DeepMD/MACE)
```

### 2. 势函数验证测试

```lammps
# 测试1: 晶格常数优化
variable a equal 3.6
lattice fcc $a
region box block 0 4 0 4 0 4

clear
read_data data.lmp

# 能量vs体积曲线
variable vol equal vol
variable pe equal pe
fix 1 all npt temp 0.01 0.01 1.0 iso 0.0 0.0 10.0

# 弹性常数计算
variable c11 equal...
```

### 3. 常见材料推荐势函数

| 材料 | 推荐势函数 | 来源 |
|-----|-----------|------|
| Cu | Mishin Cu EAM | PRB 63, 224106 (2001) |
| Al | Mishin Al EAM | PRB 65, 224114 (2002) |
| Fe | Mendelev Fe-2 EAM | PRB 79, 144106 (2009) |
| Ni | Mishin Ni EAM | PRB 59, 3393 (1999) |
| Si | Tersoff Si | PRB 38, 9902 (1988) |
| SiC | Tersoff SiC | PRB 39, 5566 (1989) |
| Ti | MEAM Ti | PRB 68, 144112 (2003) |
| H₂O | SPC/E, TIP4P/2005 | JCP 103, 3358 (1995) |
| CH₄/烷烃 | TraPPE | JPCB 102, 2569 (1998) |
| 含能材料 | ReaxFF CHON | JPC A 105, 9396 (2001) |

### 4. 势函数文件路径管理

```bash
# 设置环境变量
export LAMMPS_POTENTIALS=/path/to/potentials

# input脚本中使用环境变量
pair_coeff * * ${LAMMPS_POTENTIALS}/Cu_u3.eam.alloy Cu

# 常用势函数仓库
# 1. NIST Interatomic Potentials
#    https://www.ctcms.nist.gov/potentials/
# 2. OpenKIM
#    https://openkim.org/
# 3. Matlantis (计算数据库)
#    https://matlantis.com/
# 4. AI4Materials (ML势数据库)
#    https://github.com/AI4Materials
```

---

## 势函数格式速查

```
EAM/alloy格式 (*.eam.alloy):
元素数 N
元素符号
原子序数 质量 晶格常数 晶格类型
Nrho drho Nr dr
嵌入函数数据...
对势数据...
电子密度数据...

MEAM库文件格式 (library.meam):
elt        lat     z       ielement     atwt
alpha      b0      b1      b2           b3
...

Tersoff格式 (*.tersoff):
元素1 元素2 元素3 m gamma lambda3 c d costheta0
n beta lambda2 B R D lambda1 A
```
