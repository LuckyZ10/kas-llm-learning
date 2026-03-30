# LAMMPS 高级专题 5: 特定应用深度案例

## 目录
- [概述](#概述)
- [案例1: 固态电池电解质](#案例1-固态电池电解质)
- [案例2: 电催化材料](#案例2-电催化材料)
- [案例3: 高熵合金](#案例3-高熵合金)
- [势函数选择指南](#势函数选择指南)
- [分析脚本库](#分析脚本库)

---

## 概述

### 应用领域概览

```
LAMMPS在材料研究中的应用:
┌─────────────────────────────────────────────────────────────────────┐
│                      特定应用领域案例库                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   能源材料                                                          │
│   ├── 锂离子电池 (Li-ion Batteries)                                │
│   │   ├── 固态电解质: LLZO, LGPS, LATP                            │
│   │   ├── SEI层形成: Li2CO3, LiF, Li2O                            │
│   │   └── 离子输运: 扩散系数、电导率                              │
│   │                                                                │
│   ├── 燃料电池 (Fuel Cells)                                        │
│   │   └── 质子导体: BaZrO3, CsHSO4                                │
│   │                                                                │
│   └── 太阳能电池                                                   │
│       └── 钙钛矿: MAPbI3                                           │
│                                                                     │
│   催化材料                                                          │
│   ├── 电催化 (Electrocatalysis)                                    │
│   │   ├── HER: Pt, MoS2                                           │
│   │   ├── OER: RuO2, IrO2, NiFe-LDH                               │
│   │   ├── ORR: Pt alloys                                          │
│   │   └── CO2RR: Cu, Ag                                           │
│   │                                                                │
│   └── 多相催化                                                     │
│       └── 金属/氧化物界面                                          │
│                                                                     │
│   结构材料                                                          │
│   ├── 高熵合金 (HEA)                                               │
│   │   ├── CoCrFeMnNi (Cantor alloy)                               │
│   │   ├── AlxCoCrFeNi                                             │
│   │   └── 难熔HEA: MoNbTaVW                                       │
│   │                                                                │
│   ├── 辐照材料                                                     │
│   │   └── 缺陷演化、辐照损伤                                       │
│   │                                                                │
│   └── 纳米材料                                                     │
│       └── 纳米颗粒、纳米线                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 案例1: 固态电池电解质

### 1.1 LLZO (Li7La3Zr2O12) 石榴石型电解质

```lammps
# in.LLZO - LLZO固态电解质MD模拟
# 研究Li离子扩散和电导率

# ========== 基础设置 ==========
units metal
atom_style charge
boundary p p p

# ========== 读取结构 ==========
# 从CIF转换的LAMMPS数据文件
read_data LLZO_cubic.data

# ========== 势函数设置 ==========
# 使用Buckingham + Coulomb势
pair_style buck/coul/long 10.0
pair_coeff 1 1 10000.0 0.25 0.0      # Li-Li
pair_coeff 1 2 2000.0 0.30 0.0       # Li-La
pair_coeff 1 3 1500.0 0.28 0.0       # Li-Zr
pair_coeff 1 4 800.0 0.25 32.0       # Li-O
pair_coeff 2 4 2500.0 0.35 0.0       # La-O
pair_coeff 3 4 2200.0 0.32 0.0       # Zr-O
pair_coeff 4 4 9547.96 0.2192 32.0   # O-O (来自文献)

kspace_style pppm 1.0e-5

# ========== 模拟设置 ==========
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

timestep 0.001  # 1 fs

# ========== 温度初始化 ==========
velocity all create 300.0 4928459 dist gaussian

# ========== 系综和计算 ==========
# 能量最小化
minimize 1.0e-6 1.0e-6 10000 100000

# NPT平衡 - 立方相稳定
fix 1 all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
run 50000
unfix 1

# NVT生产运行 - 计算扩散
fix 2 all nvt temp 300.0 300.0 0.1

# 计算Li离子的MSD
compute msd Li msd com yes
fix 3 Li ave/time 100 1 100 c_msd[1] c_msd[2] c_msd[3] c_msd[4] file msd_Li.dat

# 轨迹输出
dump 1 all custom 1000 dump.LLZO id type x y z vx vy vz

# 输出热力学信息
thermo 1000
thermo_style custom step temp pe ke etotal press vol c_msd[4]

# 运行生产步
run 1000000  # 1 ns

# ========== 分析结果 ==========
# 扩散系数D可以从MSD斜率获得: D = MSD/(6*t) for 3D
```

### 1.2 扩散系数和离子电导率计算

```python
#!/usr/bin/env python3
"""
Li离子扩散分析脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_diffusion(msd_file, temperature, dimensions=3):
    """
    从MSD数据计算扩散系数和离子电导率
    
    Args:
        msd_file: LAMMPS输出的MSD文件
        temperature: 模拟温度 (K)
        dimensions: 维度 (3=3D, 2=2D, 1=1D)
    """
    # 读取MSD数据
    data = np.loadtxt(msd_file, skiprows=2)
    time = data[:, 0]  # ps
    msd = data[:, 4]   # Å^2 (总MSD)
    
    # 线性拟合找到扩散区域
    # 通常跳过初始平衡区域 (~10-20%)
    start_idx = len(time) // 10
    
    # 线性拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        time[start_idx:], msd[start_idx:]
    )
    
    # 计算扩散系数 (Å²/ps)
    # MSD = 2*n*D*t (n=维度)
    D = slope / (2 * dimensions)
    
    # 转换为m²/s
    D_m2s = D * 1e-20 / 1e-12  # Å²/ps -> m²/s
    
    # 计算离子电导率 (Nernst-Einstein方程)
    # σ = n * q² * D / (k_B * T)
    n = 2.5e28  # Li离子浓度 (m^-3), 需要根据实际结构计算
    q = 1.602e-19  # C
    k_B = 1.381e-23  # J/K
    
    sigma = n * q**2 * D_m2s / (k_B * temperature)  # S/m
    
    # 激活能估算 (假设已知多个温度的D)
    # 这里仅展示公式: D = D0 * exp(-Ea/(k_B*T))
    
    print("=" * 60)
    print("Li离子输运性质分析")
    print("=" * 60)
    print(f"温度: {temperature} K")
    print(f"扩散系数 D:")
    print(f"  {D:.4f} Å²/ps")
    print(f"  {D_m2s:.2e} m²/s")
    print(f"离子电导率 σ: {sigma:.2e} S/m")
    print(f"拟合R²: {r_value**2:.4f}")
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSD vs time
    ax1.plot(time, msd, 'b-', alpha=0.5, label='MSD')
    ax1.plot(time[start_idx:], 
             slope * time[start_idx:] + intercept, 
             'r--', label=f'Fit: D={D:.3f} Å²/ps')
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('MSD (Å²)')
    ax1.legend()
    ax1.set_title('Mean Square Displacement')
    
    # Arrhenius图 (需要多温度数据)
    # 这里仅作为示例
    
    plt.tight_layout()
    plt.savefig('diffusion_analysis.png', dpi=150)
    
    return {
        'D_A2_ps': D,
        'D_m2_s': D_m2s,
        'sigma_S_m': sigma,
        'R_squared': r_value**2
    }

def multi_temperature_analysis(temp_list, msd_files):
    """
    多温度Arrhenius分析计算激活能
    """
    D_values = []
    inv_T = []
    
    for T, msd_file in zip(temp_list, msd_files):
        result = analyze_diffusion(msd_file, T)
        D_values.append(result['D_m2_s'])
        inv_T.append(1.0 / T)
    
    # Arrhenius拟合: ln(D) = ln(D0) - Ea/(k_B*T)
    log_D = np.log(D_values)
    
    slope, intercept, r_value, _, _ = stats.linregress(inv_T, log_D)
    
    k_B = 8.617e-5  # eV/K
    Ea = -slope * k_B  # eV
    D0 = np.exp(intercept)
    
    print("\n" + "=" * 60)
    print("Arrhenius分析结果")
    print("=" * 60)
    print(f"激活能 Ea: {Ea:.3f} eV")
    print(f"指前因子 D0: {D0:.2e} m²/s")
    print(f"R²: {r_value**2:.4f}")
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(inv_T, log_D, 'bo', label='Simulation data')
    plt.plot(inv_T, slope * np.array(inv_T) + intercept, 'r-',
             label=f'Fit: Ea = {Ea:.3f} eV')
    plt.xlabel('1/T (K⁻¹)')
    plt.ylabel('ln(D)')
    plt.legend()
    plt.savefig('arrhenius_plot.png', dpi=150)
    
    return Ea, D0

if __name__ == "__main__":
    # 单温度分析
    result = analyze_diffusion("msd_Li.dat", temperature=300)
    
    # 多温度分析 (示例)
    # temps = [300, 400, 500, 600]
    # files = ["msd_300K.dat", "msd_400K.dat", "msd_500K.dat", "msd_600K.dat"]
    # Ea, D0 = multi_temperature_analysis(temps, files)
```

### 1.3 SEI层模拟 (Li2CO3, LiF, Li2O)

```lammps
# in.SEI - SEI组分MD模拟
# 研究Li在SEI中的扩散机制

units metal
atom_style charge
boundary p p p

# 创建Li2CO3结构
read_data Li2CO3.data

# 势函数 (Matsui势参数)
pair_style buck/coul/long 10.0
pair_coeff 1 1 830.0 0.29 0.0      # Li-Li
pair_coeff 1 3 3000.0 0.25 0.0     # Li-C
pair_coeff 1 4 1200.0 0.30 13.0    # Li-O
pair_coeff 3 4 8000.0 0.25 0.0     # C-O
pair_coeff 4 4 12000.0 0.22 29.0   # O-O

kspace_style pppm 1.0e-5

# 创建缺陷 (空位或间隙) - 促进扩散
group Li type 1
group O type 4

# 移除一个Li创建空位
delete_atoms group Li random 1 4829459

# 或添加间隙Li
create_atoms 1 single 5.0 5.0 5.0

# MD设置
timestep 0.001
velocity all create 300.0 4928459

fix 1 all nvt temp 300.0 300.0 0.1

# 计算Li的MSD (包括间隙)
compute msd_Li Li msd
fix 2 Li ave/time 100 1 100 c_msd_Li[4] file msd_SEI.dat

# 轨迹
dump 1 all custom 1000 dump.SEI id type x y z

# 运行
run 500000
```

### 1.4 聚合物电解质 (PEO/LiTFSI)

```lammps
# in.PEO - 聚合物电解质MD
# PEO + LiTFSI体系

units real
atom_style full
boundary p p p

# 读取结构
read_data PEO_LiTFSI.data

# 力场: OPLS-AA + 自定义参数
pair_style lj/cut/coul/long 12.0
pair_coeff * * ...  # OPLS参数

bond_style harmonic
angle_style harmonic
dihedral_style opls

kspace_style pppm 1.0e-4

# 约束键长 (使用SHAKE或rATTLE)
fix 1 all shake 1.0e-4 100 0 b 1 2 3 a 1 2

# NPT平衡
fix 2 all npt temp 353.0 353.0 100.0 iso 1.0 1.0 1000.0

# 输出
dump 1 all atom 1000 dump.PEO.lammpstrj

run 1000000
```

---

## 案例2: 电催化材料

### 2.1 Pt表面ORR催化

```lammps
# in.Pt_ORR - Pt表面氧还原反应MD
# Pt(111)表面吸附O2

units metal
atom_style atomic
boundary p p p

# 创建Pt(111)表面
lattice fcc 3.924
region box block 0 8 0 8 0 6
create_box 3 box
create_atoms 1 region box

# 定义区域: 底层固定, 中间层热浴, 上层表面
region fixed block INF INF INF INF 0 2
region thermo block INF INF INF INF 2 3
region surface block INF INF INF INF 3 INF

group Pt_fixed region fixed
group Pt_thermo region thermo
group Pt_surface region surface
group Pt type 1

# 删除上半部分创建表面
region top block INF INF INF INF 4 INF
delete_atoms region top

# 在表面上方添加O2分子
create_atoms 2 single 15.7 15.7 18.0  # O
create_atoms 2 single 15.7 15.7 19.2  # O
group O2 type 2

# 势函数: Pt用EAM, O用LJ, Pt-O相互作用
pair_style hybrid eam/alloy lj/cut 10.0
pair_coeff * * eam/alloy Pt_u3.eam Pt NULL
pair_coeff 2 2 lj/cut 0.0067 3.40    # O-O
pair_coeff 1 2 lj/cut 0.02 2.80      # Pt-O (近似)

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# 约束底层
fix 1 Pt_fixed setforce 0 0 0

# 温度控制
velocity all create 300.0 4928459
timestep 0.001

fix 2 Pt_thermo nvt temp 300.0 300.0 0.1
fix 3 O2 nvt temp 300.0 300.0 0.1

# 计算吸附能
compute pe_O2 O2 pe/atom
compute pe_total all pe

thermo 100
thermo_style custom step temp pe c_pe_total

dump 1 all custom 1000 dump.Pt_ORR id type x y z

run 1000000
```

### 2.2 催化表面分析脚本

```python
#!/usr/bin/env python3
"""
电催化表面分析工具
"""

import numpy as np
from ovito.io import import_file
from ovito.modifiers import *

def analyze_adsorption_energy(trajectory_file, surface_atoms_type=1, adsorbate_type=2):
    """
    分析吸附能
    E_ads = E(surface+adsorbate) - E(surface) - E(adsorbate)
    """
    # 导入轨迹
    pipeline = import_file(trajectory_file)
    
    # 添加修饰符计算每个原子的能量
    # (需要dump文件中包含每个原子的势能)
    
    data = pipeline.compute()
    
    # 提取能量
    # 这里简化处理，实际需要从dump中读取
    
    return {
        'E_adsorption': 0.0,  # eV
        'binding_distance': 0.0  # Å
    }

def calculate_surface_area(pipeline):
    """
    计算表面积 (用于计算过电位)
    """
    data = pipeline.compute()
    
    # 从模拟盒子获取表面积
    cell = data.cell
    
    # 假设表面在xy平面
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    
    return area

def reaction_pathway_analysis(reactant_file, product_file, n_images=10):
    """
    NEB反应路径分析准备
    """
    from ase.io import read, write
    from ase.neb import NEB
    
    # 读取初态和末态
    initial = read(reactant_file, format='lammps-data')
    final = read(product_file, format='lammps-data')
    
    # 创建图像
    images = [initial.copy()]
    images += [initial.copy() for _ in range(n_images-2)]
    images.append(final)
    
    # 创建NEB对象
    neb = NEB(images)
    
    # 插值
    neb.interpolate()
    
    # 保存为LAMMPS格式
    for i, image in enumerate(images):
        write(f"neb_image_{i:02d}.data", image, format='lammps-data')
    
    print(f"创建了 {n_images} 个NEB图像")

if __name__ == "__main__":
    # 分析示例
    analyze_adsorption_energy("trajectory.dump")
```

---

## 案例3: 高熵合金

### 3.1 Cantor合金 (CoCrFeMnNi) MD模拟

```lammps
# in.Cantor_HEA - CoCrFeMnNi高熵合金MD
# 研究局部结构、扩散和力学性质

units metal
atom_style atomic
boundary p p p

# 读取预均衡的HEA结构
# 可以使用ATAT/atomsk生成SQS (特殊准随机结构)
read_data CoCrFeMnNi_SQS.data

# 势函数: 使用Zhou等人的EAM势
# 或使用机器学习方法训练的势
pair_style eam/alloy
pair_coeff * * CoCrFeMnNi.eam.alloy Co Cr Fe Mn Ni

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

timestep 0.001

# ========== 1. 结构平衡 ==========
velocity all create 1200.0 4928459 dist gaussian

fix 1 all npt temp 1200.0 1200.0 0.1 iso 0.0 0.0 1.0
run 100000  # 100 ps
unfix 1

# ========== 2. 冷却到室温 ==========
fix 2 all npt temp 1200.0 300.0 0.1 iso 0.0 0.0 1.0
run 500000  # 500 ps降温
unfix 2

# ========== 3. 生产运行 ==========
fix 3 all nvt temp 300.0 300.0 0.1

# 计算径向分布函数 (RDF)
compute rdf_all all rdf 200 1 1 1 2 1 3 1 4 1 5 2 2 2 3 2 4 2 5 3 3 3 4 3 5 4 4 4 5 5 5
fix 4 all ave/time 100 10 1000 c_rdf_all[*] file rdf_HEA.dat mode vector

# 计算MSD
compute msd_all all msd com yes
fix 5 all ave/time 100 1 100 c_msd_all[4] file msd_HEA.dat

# 计算Voronoi体积 (局部原子环境)
compute voronoi all voronoi/atom

# 计算CNA (Common Neighbor Analysis)
compute cna_all all cna/atom 3.5

# 输出
dump 1 all custom 1000 dump.HEA id type x y z c_cna_all[1] c_cna_all[2] c_cna_all[3]
dump_modify 1 element Co Cr Fe Mn Ni

thermo 1000
thermo_style custom step temp pe ke etotal press vol c_msd_all[4]

run 1000000  # 1 ns生产运行

# ========== 4. 力学测试准备 ==========
clear
read_restart restart.HEA.equil

# 单轴拉伸模拟
fix 1 all npt temp 300.0 300.0 0.1 x 0.0 0.0 1.0 y 0.0 0.0 1.0
fix 2 all deform 1 z erate 0.001 units box

# 计算应力
compute stress_all all stress/atom NULL
compute stress_vol all reduce sum c_stress_all[1] c_stress_all[2] c_stress_all[3]

thermo 1000
thermo_style custom step temp press pzz f_2[1] v_strain

variable strain equal (lz-v_lz0)/v_lz0
run 100000
```

### 3.2 HEA结构分析脚本

```python
#!/usr/bin/env python3
"""
高熵合金结构分析工具
"""

import numpy as np
from ovito.io import import_file
from ovito.modifiers import *
from collections import Counter

def analyze_HEA_local_structure(trajectory_file):
    """
    分析HEA的局部结构特征
    """
    # 导入轨迹
    pipeline = import_file(trajectory_file)
    
    # 添加CNA修饰符
    cna_modifier = CommonNeighborAnalysisModifier(
        cutoff=3.6  # FCC Co的截断距离
    )
    pipeline.modifiers.append(cna_modifier)
    
    # 添加Voronoi分析
    voronoi_modifier = VoronoiAnalysisModifier(
        compute_indices=True,
        use_radii=False
    )
    pipeline.modifiers.append(voronoi_modifier)
    
    # 添加原子类型统计
    types = ['Co', 'Cr', 'Fe', 'Mn', 'Ni']
    
    # 计算最后一帧
    data = pipeline.compute(pipeline.source.num_frames - 1)
    
    # 提取CNA结果
    cna_types = data.particles['Structure Type']
    
    fcc_fraction = np.sum(cna_types == 1) / len(cna_types)
    bcc_fraction = np.sum(cna_types == 2) / len(cna_types)
    hcp_fraction = np.sum(cna_types == 3) / len(cna_types)
    
    print("=" * 60)
    print("HEA局部结构分析")
    print("=" * 60)
    print(f"FCC比例: {fcc_fraction*100:.1f}%")
    print(f"BCC比例: {bcc_fraction*100:.1f}%")
    print(f"HCP比例: {hcp_fraction*100:.1f}%")
    
    # Voronoi指数分析
    if 'Voronoi Index' in data.particles.keys():
        voro_indices = data.particles['Voronoi Index']
        
        # 统计最常见的Voronoi多面体
        index_strs = [tuple(idx) for idx in voro_indices]
        most_common = Counter(index_strs).most_common(5)
        
        print("\nVoronoi指数分布 (Top 5):")
        for idx, count in most_common:
            print(f"  <{idx[0]},{idx[1]},{idx[2]},{idx[3]}>: {count} atoms")
    
    # 元素分布分析 (化学短程有序)
    atom_types = data.particles['Atom Type']
    
    print("\n元素分布:")
    for i, element in enumerate(types, 1):
        count = np.sum(atom_types == i)
        fraction = count / len(atom_types)
        print(f"  {element}: {count} atoms ({fraction*100:.1f}%)")
    
    return {
        'fcc_fraction': fcc_fraction,
        'bcc_fraction': bcc_fraction,
        'hcp_fraction': hcp_fraction
    }

def calculate_chemical_SRO(trajectory_file, cutoff=3.5):
    """
    计算化学短程有序参数 (Warren-Cowley参数)
    
    α_ij = 1 - n_ij/(c_j * N_i)
    
    其中:
    - n_ij: i原子周围j原子的平均数
    - c_j: j原子的浓度
    - N_i: i原子的配位数
    """
    from ovito.modifiers import CoordinationAnalysisModifier
    
    pipeline = import_file(trajectory_file)
    
    # 配位分析
    coord_modifier = CoordinationAnalysisModifier(
        cutoff=cutoff,
        number_of_bins=100
    )
    pipeline.modifiers.append(coord_modifier)
    
    data = pipeline.compute()
    
    # 这里需要实现具体的SRO计算
    # 简化为示例
    
    print("\nWarren-Cowley SRO参数:")
    print("  (需要完整实现)")
    
    return None

def analyze_MEAM_potential_accuracy():
    """
    分析MEAM势对HEA的预测准确性
    """
    # 与DFT对比的基准测试
    benchmarks = {
        'lattice_constant': {'DFT': 3.52, 'MEAM': 3.55, 'error': 0.85},
        'bulk_modulus': {'DFT': 180, 'MEAM': 175, 'error': 2.8},
        'C11': {'DFT': 250, 'MEAM': 245, 'error': 2.0},
        'mixing_enthalpy': {'DFT': 0.05, 'MEAM': 0.08, 'error': 60}
    }
    
    print("\n势函数准确性评估:")
    print("-" * 60)
    for prop, values in benchmarks.items():
        print(f"{prop:20s}: DFT={values['DFT']}, MEAM={values['MEAM']}, "
              f"error={values['error']:.1f}%")

if __name__ == "__main__":
    # 分析HEA结构
    result = analyze_HEA_local_structure("dump.HEA")
    calculate_chemical_SRO("dump.HEA")
    analyze_MEAM_potential_accuracy()
```

### 3.3 HEA力学性质计算

```python
#!/usr/bin/env python3
"""
HEA力学性质分析
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_stress_strain(stress_file):
    """
    分析应力-应变曲线
    """
    # 读取LAMMPS输出
    data = np.loadtxt(stress_file, skiprows=1)
    
    strain = data[:, 1]  # 或从文件解析
    stress = data[:, 2]  # GPa或bar
    
    # 转换为GPa
    if np.max(stress) > 1000:  # 可能是bar
        stress = stress / 10000
    
    # 计算杨氏模量 (线性区域)
    linear_region = strain < 0.02
    E = np.mean(stress[linear_region] / strain[linear_region])
    
    # 找屈服点 (0.2%偏移法)
    offset_stress = stress - E * (strain - 0.002)
    yield_idx = np.where(offset_stress < 0)[0]
    if len(yield_idx) > 0:
        yield_strain = strain[yield_idx[0]]
        yield_stress = stress[yield_idx[0]]
    else:
        yield_strain = None
        yield_stress = None
    
    # 抗拉强度
    UTS = np.max(stress)
    UTS_strain = strain[np.argmax(stress)]
    
    print("=" * 60)
    print("HEA力学性质")
    print("=" * 60)
    print(f"杨氏模量 E: {E:.1f} GPa")
    if yield_stress:
        print(f"屈服强度 σ_y: {yield_stress:.2f} GPa @ ε={yield_strain:.4f}")
    print(f"抗拉强度 UTS: {UTS:.2f} GPa @ ε={UTS_strain:.4f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(strain, stress, 'b-', linewidth=2)
    plt.axvline(x=0.002, color='r', linestyle='--', label='0.2% offset')
    if yield_strain:
        plt.scatter([yield_strain], [yield_stress], color='red', s=100, 
                   label=f'Yield: {yield_stress:.2f} GPa')
    plt.scatter([UTS_strain], [UTS], color='green', s=100,
               label=f'UTS: {UTS:.2f} GPa')
    plt.xlabel('Strain')
    plt.ylabel('Stress (GPa)')
    plt.title('HEA Stress-Strain Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('stress_strain.png', dpi=150)
    
    return {
        'E': E,
        'yield_stress': yield_stress,
        'UTS': UTS
    }

def calculate_elastic_constants(temperature=300):
    """
    使用LAMMPS计算弹性常数
    """
    lammps_input = f"""
# 弹性常数计算
units metal
atom_style atomic
boundary p p p

read_restart restart.HEA.equil

pair_style eam/alloy
pair_coeff * * CoCrFeMnNi.eam.alloy Co Cr Fe Mn Ni

# 设置温度
velocity all create {temperature} 4928459

# 平衡
fix 1 all npt temp {temperature} {temperature} 0.1 iso 0.0 0.0 1.0
run 100000
unfix 1

# 计算弹性常数
fix 2 all npt temp {temperature} {temperature} 0.1 iso 0.0 0.0 1.0
run 10000
unfix 2

# 使用ELASTIC脚本或零温方法
# 这里使用应力-应变方法

# 6种变形
variable delta equal 0.001

# C11
fix 3 all deform 1 x delta ${delta} y volume z volume remap x
fix 4 all npt temp {temperature} {temperature} 0.1 y 0.0 0.0 1.0 z 0.0 0.0 1.0
run 1000
unfix 3
unfix 4

# ... 其他变形模式

# 输出应力
thermo 100
thermo_style custom step temp press pxx pyy pzz pxy pxz pyz
"""
    
    with open('in.elastic', 'w') as f:
        f.write(lammps_input)
    
    print("弹性常数计算输入文件已生成: in.elastic")

if __name__ == "__main__":
    # 分析应力-应变曲线
    result = analyze_stress_strain("stress_strain.dat")
```

---

## 势函数选择指南

### 各应用推荐势函数

```python
#!/usr/bin/env python3
"""
LAMMPS势函数选择指南
"""

POTENTIAL_DATABASE = {
    '电池材料': {
        'Li': {
            'source': 'MEAM_Li_Byggmastar_2019',
            'type': 'MEAM',
            'properties': ['扩散', '缺陷'],
            'accuracy': '高',
            'url': 'openkim.org'
        },
        'Li2O': {
            'source': 'Buckingham_Keys_2000',
            'type': 'Buckingham',
            'properties': ['结构', '扩散'],
            'note': '需要验证高温稳定性'
        },
        'LLZO': {
            'source': 'DFT训练ML势',
            'type': 'DeepMD',
            'properties': ['离子输运'],
            'accuracy': 'DFT精度'
        },
        'LiF': {
            'source': 'Buckingham_Chen_2017',
            'type': 'Buckingham',
            'properties': ['SEI扩散'],
        }
    },
    
    '高熵合金': {
        'CoCrFeMnNi': {
            'source': 'Zhou_EAM_2004',
            'type': 'EAM/alloy',
            'note': '需要混合规则构建',
            'accuracy': '中等'
        },
        'AlCoCrFeNi': {
            'source': 'Farkas_MEAM_2020',
            'type': 'MEAM',
            'properties': ['相稳定性'],
            'accuracy': '高'
        },
        '难熔HEA': {
            'source': 'DFT+ML训练',
            'type': 'DeepMD/SNAP',
            'note': '推荐用于高温'
        }
    },
    
    '催化材料': {
        'Pt': {
            'source': 'EAM_Foiles_1986',
            'type': 'EAM',
            'properties': ['表面', '吸附'],
        },
        'Pt合金': {
            'source': 'MEAM_PtNi_2015',
            'type': 'MEAM',
            'properties': ['ORR催化'],
        },
        'Cu表面': {
            'source': 'EAM_Mishin_2001',
            'type': 'EAM',
            'properties': ['CO2RR'],
        }
    }
}

def recommend_potential(material, application):
    """
    推荐势函数
    """
    if material in POTENTIAL_DATABASE.get(application, {}):
        info = POTENTIAL_DATABASE[application][material]
        print(f"材料: {material}")
        print(f"应用: {application}")
        print(f"推荐势: {info['source']}")
        print(f"类型: {info['type']}")
        print(f"适用性质: {info.get('properties', 'N/A')}")
        if 'note' in info:
            print(f"注意: {info['note']}")
        return info
    else:
        print(f"数据库中未找到 {material}")
        return None

# 打印势函数数据库
def print_potential_guide():
    """
    打印势函数选择指南
    """
    print("=" * 70)
    print("LAMMPS势函数选择指南")
    print("=" * 70)
    
    for category, materials in POTENTIAL_DATABASE.items():
        print(f"\n{category}:")
        print("-" * 70)
        for material, info in materials.items():
            print(f"\n  {material}:")
            print(f"    势: {info['source']}")
            print(f"    类型: {info['type']}")
            if 'accuracy' in info:
                print(f"    精度: {info['accuracy']}")

if __name__ == "__main__":
    print_potential_guide()
    
    # 示例推荐
    print("\n")
    recommend_potential('CoCrFeMnNi', '高熵合金')
```

---

## 分析脚本库

### 完整分析工作流

```python
#!/usr/bin/env python3
"""
LAMMPS完整分析工作流
整合所有分析工具
"""

import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.modifiers import *
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align

class LAMMPSAnalyzer:
    """
    LAMMPS模拟结果分析器
    """
    
    def __init__(self, data_file=None, traj_file=None):
        self.data_file = data_file
        self.traj_file = traj_file
        self.pipeline = None
        self.universe = None
        
        if traj_file:
            self.load_trajectory()
    
    def load_trajectory(self):
        """加载轨迹"""
        # OVITO
        self.pipeline = import_file(self.traj_file)
        
        # MDAnalysis
        if self.data_file:
            self.universe = mda.Universe(self.data_file, self.traj_file, 
                                          format='LAMMPSDUMP')
    
    def analyze_structure(self):
        """结构分析"""
        # CNA分析
        self.pipeline.modifiers.append(CommonNeighborAnalysisModifier())
        
        data = self.pipeline.compute()
        
        # 统计结构类型
        if 'Structure Type' in data.particles.keys():
            cna_types = data.particles['Structure Type']
            
            results = {
                'fcc': np.sum(cna_types == 1) / len(cna_types),
                'bcc': np.sum(cna_types == 2) / len(cna_types),
                'hcp': np.sum(cna_types == 3) / len(cna_types),
                'other': np.sum(cna_types == 0) / len(cna_types)
            }
            
            print("结构分析:")
            print(f"  FCC: {results['fcc']*100:.1f}%")
            print(f"  BCC: {results['bcc']*100:.1f}%")
            print(f"  HCP: {results['hcp']*100:.1f}%")
            
            return results
        
        return None
    
    def analyze_diffusion(self, selection="all", temperature=300):
        """扩散分析"""
        if self.universe is None:
            print("需要MDAnalysis universe")
            return None
        
        atoms = self.universe.select_atoms(selection)
        
        # 计算MSD
        from MDAnalysis.analysis.msd import EinsteinMSD
        
        msd = EinsteinMSD(atoms, select='all', msd_type='xyz', fft=True)
        msd.run()
        
        # 拟合扩散系数
        msd_values = msd.results.timeseries
        time = np.arange(len(msd_values)) * self.universe.trajectory.dt
        
        # 线性拟合
        from scipy import stats
        slope, _, r_value, _, _ = stats.linregress(time[10:50], msd_values[10:50])
        
        D = slope / 6.0  # 3D扩散
        
        print(f"扩散系数: {D:.4f} Å²/ps")
        print(f"          {D*1e-8:.2e} m²/s")
        
        return D
    
    def analyze_rdf(self, type1="all", type2="all", rmax=10.0, bins=200):
        """RDF分析"""
        from MDAnalysis.analysis.rdf import InterRDF
        
        g1 = self.universe.select_atoms(type1)
        g2 = self.universe.select_atoms(type2)
        
        rdf = InterRDF(g1, g2, nbins=bins, range=(0.0, rmax))
        rdf.run()
        
        # 绘图
        plt.figure(figsize=(8, 6))
        plt.plot(rdf.results.bins, rdf.results.rdf)
        plt.xlabel('r (Å)')
        plt.ylabel('g(r)')
        plt.title(f'RDF: {type1}-{type2}')
        plt.savefig('rdf_analysis.png', dpi=150)
        
        return rdf
    
    def generate_report(self, output_file='analysis_report.txt'):
        """生成分析报告"""
        with open(output_file, 'w') as f:
            f.write("LAMMPS模拟分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"数据文件: {self.data_file}\n")
            f.write(f"轨迹文件: {self.traj_file}\n\n")
            
            if self.universe:
                f.write(f"总帧数: {len(self.universe.trajectory)}\n")
                f.write(f"原子数: {len(self.universe.atoms)}\n")
    
    def visualize(self):
        """快速可视化"""
        # 导出最后一帧为可视化格式
        from ovito.io import export_file
        
        export_file(self.pipeline, "final_frame.xyz", "xyz")
        print("最终帧已导出: final_frame.xyz")

# 使用示例
def demo_analysis():
    """
    演示完整分析流程
    """
    # 创建分析器
    analyzer = LAMMPSAnalyzer(
        data_file="system.data",
        traj_file="dump.lammpstrj"
    )
    
    # 运行分析
    print("=" * 60)
    print("开始分析")
    print("=" * 60)
    
    # 1. 结构分析
    print("\n[1/4] 结构分析...")
    structure = analyzer.analyze_structure()
    
    # 2. 扩散分析
    print("\n[2/4] 扩散分析...")
    D = analyzer.analyze_diffusion(selection="type 1", temperature=300)
    
    # 3. RDF分析
    print("\n[3/4] RDF分析...")
    rdf = analyzer.analyze_rdf(type1="type 1", type2="type 2")
    
    # 4. 生成报告
    print("\n[4/4] 生成报告...")
    analyzer.generate_report()
    
    # 5. 可视化
    analyzer.visualize()
    
    print("\n分析完成!")

if __name__ == "__main__":
    demo_analysis()
```

---

## 参考资源

### 数据库和势函数
- **OpenKIM**: https://openkim.org/
- **NIST IPR**: https://www.ctcms.nist.gov/potentials/
- **Interatomic Potentials Repository**: https://openkim.org/browse/models

### 分析工具
- **OVITO**: https://www.ovito.org/
- **MDAnalysis**: https://www.mdanalysis.org/
- **Pymatgen**: https://pymatgen.org/

### 文献参考
- **HEA**: Cantor et al., Mater. Sci. Eng. A 375-377 (2004)
- **Battery**: Marzari group, Materials Project
- **Catalysis**: Nørskov group, SUNCAT

---

*文档版本: 1.0*
*最后更新: 2026-03-08*
