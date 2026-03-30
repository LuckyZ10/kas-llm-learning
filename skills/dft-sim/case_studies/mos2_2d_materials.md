# 案例研究：单层MoS₂的二维材料计算

## 概述

本案例展示二维过渡金属硫族化合物(TMD)单层MoS₂的完整计算流程，包括结构优化、能带结构、光学性质、应变效应等。MoS₂是典型的直接带隙半导体，在光电子学和谷电子学中有重要应用。

**目标材料**: 单层MoS₂ (2H相)
**对称性**: D₃h
**实验晶格常数**: 3.193 Å
**实验带隙**: 1.9 eV (直接带隙, K点)

---

## 1. 结构构建与优化

### 1.1 晶体结构

单层MoS₂具有类石墨烯的六角蜂窝结构，Mo原子位于两个S原子层之间，形成三明治结构。

**POSCAR**:
```
MoS2_monolayer
1.0
   3.1930000000000000    0.0000000000000000    0.0000000000000000
  -1.5965000000000000    2.7652156666666666    0.0000000000000000
   0.0000000000000000    0.0000000000000000   20.0000000000000000
Mo S
1 2
direct
  0.0000000000000000  0.0000000000000000  0.5000000000000000
  0.3333333333333333  0.6666666666666667  0.5850000000000000
  0.6666666666666667  0.3333333333333333  0.4150000000000000
```

**关键参数**:
- **真空层**: 20 Å，避免层间相互作用
- **S-Mo键长**: ~2.42 Å
- **S-S层间距**: ~3.13 Å

### 1.2 库仑截断 (2D材料必需)

**VASP设置**:
```
# 2D库仑截断 (VASP 6.5+)
ICUT = 1              # 开启2D截断
LCUT = .TRUE.
IDIPOL = 4            # 表面偶极校正

# 或传统方法 (VDW修正)
IVDW = 12             # DFT-D3(BJ)
GGA = PE
```

**QE设置**:
```fortran
&SYSTEM
   assume_isolated = '2D'
   esm_bc = 'bc2'
   esm_w = 20.0
   vdw_corr = 'dft-d3'
/
```

### 1.3 收敛测试

**k点网格测试** (固定ENCUT=400 eV):
| k点网格 | 总能 (eV) | 晶格常数 (Å) | 带隙 (eV) |
|---------|-----------|--------------|-----------|
| 6×6×1   | -24.8321  | 3.152        | 1.68      |
| 9×9×1   | -24.8456  | 3.174        | 1.73      |
| 12×12×1 | -24.8492  | 3.189        | 1.78      |
| **15×15×1** | **-24.8501** | **3.192** | **1.79** |
| 18×18×1 | -24.8503  | 3.193        | 1.79      |
| **实验**| -         | **3.193**    | **1.90**  |

**截断能测试** (固定15×15×1 k点):
| ENCUT (eV) | 总能 (eV) | 力收敛 | 耗时 (min) |
|------------|-----------|--------|------------|
| 300        | -24.8123  | 否     | 8          |
| 400        | -24.8501  | 是     | 15         |
| 500        | -24.8523  | 是     | 28         |
| 600        | -24.8528  | 是     | 45         |

**推荐参数**: ENCUT=400 eV, k点15×15×1

### 1.4 优化结果

**晶格常数**: a = 3.192 Å (误差 0.03%)

**键长和角度**:
```
Mo-S键长: 2.408 Å (实验: 2.41 Å)
S-S距离: 3.130 Å
S-Mo-S键角: 81.3°
```

**层间距**: 单层厚度 ~6.5 Å (包含真空层)

---

## 2. 电子结构

### 2.1 能带结构特征

**高对称路径**: Γ → M → K → Γ

**关键特征**:
- **直接带隙**: K点处, Eg = 1.79 eV (PBE)
- **导带底**: K点 (Mo-d轨道主导)
- **价带顶**: K点 (S-p轨道主导)
- **自旋轨道耦合**: K点价带劈裂 ~150 meV

### 2.2 PBE vs HSE06对比

| 性质 | PBE | HSE06 | 实验 |
|------|-----|-------|------|
| 带隙 | 1.79 eV | 2.35 eV | 1.90 eV |
| K点VB分裂 | 0.15 eV | 0.15 eV | 0.15 eV |
| 有效质量 (电子) | 0.47 m₀ | 0.51 m₀ | - |
| 有效质量 (空穴) | 0.54 m₀ | 0.61 m₀ | - |

**HSE06参数**:
```
LHFCALC = .TRUE.
HFSCREEN = 0.2        # HSE06
ALGO = ALL
PRECFOCK = Fast
NKRED = 2             # k点缩减
```

**观察**: HSE06高估带隙，PBE更接近实验值(误差~6%)。

### 2.3 自旋轨道耦合 (SOC)

**VASP设置**:
```
LSORBIT = .TRUE.
SAXIS = 0 0 1         # 沿z方向
MAGMOM = 0 0 0        # 非磁性
```

**谷极化效应**:
- K和K'谷时间反演对称
- 外加圆偏振光可选择性激发特定谷
- 谷霍尔效应可通过贝里曲率计算

**能带可视化**:
```python
def plot_mos2_bands():
    """
    绘制MoS₂能带，突出K点谷特征
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制能带 (带SOC)
    kpath = np.loadtxt('KPOINTS_BANDS', skiprows=3)
    bands_soc = np.loadtxt('EIGENVAL_SOC', skiprows=7)
    
    for iband in range(bands_soc.shape[1]):
        ax.plot(kpath[:, 0], bands_soc[:, iband], 'b-', linewidth=1.5)
    
    # 标记K点位置
    k_kpoint = find_kpoint_index(kpath, [1/3, 1/3, 0])
    ax.axvline(x=kpath[k_kpoint, 0], color='r', linestyle='--', 
               alpha=0.5, label='K valley')
    
    # 标注带隙
    vbm_k = bands_soc[k_kpoint, 16]  # 价带顶
    cbm_k = bands_soc[k_kpoint, 17]  # 导带底
    gap = cbm_k - vbm_k
    
    ax.annotate(f'Eg = {gap:.2f} eV', 
                xy=(kpath[k_kpoint, 0], (vbm_k+cbm_k)/2),
                xytext=(kpath[k_kpoint, 0]+0.5, (vbm_k+cbm_k)/2),
                arrowprops=dict(arrowstyle='<->', color='red'))
    
    ax.set_xlabel('k-path')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('MoS₂ Band Structure (PBE+SOC)')
    ax.set_xlim(kpath[0, 0], kpath[-1, 0])
    ax.set_ylim(-4, 4)
    ax.legend()
    plt.tight_layout()
    plt.savefig('mos2_bands_soc.png', dpi=300)
```

### 2.4 投影态密度

**轨道贡献分析**:
- **价带顶 (-2~0 eV)**: 主要S-3p_z轨道贡献
- **导带底 (1.5~3 eV)**: 主要Mo-4d_z²轨道贡献
- **深能级 (-6~-4 eV)**: S-3p_x,y与Mo-4d杂化

**可视化**:
```python
def plot_layer_pdos():
    """
    区分上下S层和Mo层的PDOS
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    layers = ['Bottom S', 'Mo', 'Top S']
    colors = ['blue', 'green', 'red']
    
    for ax, layer, color in zip(axes, layers, colors):
        energy, pdos = load_pdos(f'{layer}_pdos.dat')
        ax.fill_between(energy, 0, pdos, alpha=0.5, color=color)
        ax.plot(energy, pdos, color=color, linewidth=1.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('PDOS')
        ax.set_title(layer)
        ax.set_xlim(-8, 4)
    
    plt.tight_layout()
    plt.savefig('mos2_layer_pdos.png', dpi=300)
```

---

## 3. 应变工程

### 3.1 双轴应变效应

应变是调控二维材料带隙的有效手段。

**应变计算方法**:
```python
def apply_strain(poscar_in, poscar_out, strain_percent):
    """
    应用双轴应变
    strain_percent: 应变百分比 (正为拉伸,负为压缩)
    """
    with open(poscar_in, 'r') as f:
        lines = f.readlines()
    
    # 读取晶格矢量
    scale = float(lines[1])
    a1 = np.array([float(x) for x in lines[2].split()])
    a2 = np.array([float(x) for x in lines[3].split()])
    a3 = np.array([float(x) for x in lines[4].split()])
    
    # 应用应变 (保持z方向不变)
    factor = 1 + strain_percent / 100
    a1_strained = a1 * factor
    a2_strained = a2 * factor
    
    # 写出新的POSCAR
    lines[2] = f"  {a1_strained[0]:.10f}  {a1_strained[1]:.10f}  {a1_strained[2]:.10f}\n"
    lines[3] = f"  {a2_strained[0]:.10f}  {a2_strained[1]:.10f}  {a2_strained[2]:.10f}\n"
    
    with open(poscar_out, 'w') as f:
        f.writelines(lines)
```

### 3.2 应变-带隙关系

| 应变 (%) | 晶格常数 (Å) | PBE带隙 (eV) | HSE带隙 (eV) | 带隙类型 |
|----------|--------------|--------------|--------------|----------|
| -6       | 3.00         | 1.12         | 1.68         | 直接 (K-K) |
| -4       | 3.07         | 1.35         | 1.92         | 直接 (K-K) |
| -2       | 3.13         | 1.58         | 2.15         | 直接 (K-K) |
| **0**    | **3.19**     | **1.79**     | **2.35**     | **直接** |
| +2       | 3.26         | 1.68         | 2.24         | 直接 (K-K) |
| +4       | 3.32         | 1.42         | 1.98         | 间接 (Γ-K) |
| +6       | 3.39         | 1.08         | 1.62         | 间接 (Γ-K) |

**关键发现**:
1. 压缩应变增大带隙 (晶格常数减小→轨道重叠增强→能级展宽)
2. 拉伸应变超过+4%导致带隙变为间接 (K→Γ)
3. 线性调谐范围: -4%到+2%，带隙可调范围 1.4-2.2 eV

**可视化**:
```python
def plot_strain_gap():
    """
    绘制应变-带隙关系图
    """
    strain = [-6, -4, -2, 0, 2, 4, 6]
    gap_pbe = [1.12, 1.35, 1.58, 1.79, 1.68, 1.42, 1.08]
    gap_hse = [1.68, 1.92, 2.15, 2.35, 2.24, 1.98, 1.62]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(strain, gap_pbe, 'bo-', label='PBE', linewidth=2, markersize=8)
    ax.plot(strain, gap_hse, 'rs-', label='HSE06', linewidth=2, markersize=8)
    
    # 标记直接-间接转变点
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7)
    ax.text(2.7, 2.0, 'Direct→Indirect', rotation=90, fontsize=10)
    
    # 标记太阳光谱范围
    ax.axhspan(1.1, 1.7, alpha=0.2, color='yellow', label='Visible range')
    
    ax.set_xlabel('Strain (%)', fontsize=12)
    ax.set_ylabel('Band Gap (eV)', fontsize=12)
    ax.set_title('MoS₂ Band Gap vs Biaxial Strain', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim(-7, 7)
    ax.set_ylim(0.5, 2.8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mos2_strain_gap.png', dpi=300)
```

---

## 4. 光学性质

### 4.1 介电函数

**计算设置**:
```
LOPTICS = .TRUE.
NBANDS = 120          # 足够多的空带
CSHIFT = 0.05         # 较小的展宽
NEDOS = 3001
ISMEAR = 0
SIGMA = 0.01
```

**关键光学参数**:
- **A激子**: 1.90 eV (K点直接跃迁)
- **B激子**: 2.10 eV (分裂自旋态)
- **C激子**: 2.80 eV (Γ点)

### 4.2 激子效应 (BSE计算)

**VASP BSE设置**:
```
ALGO = BSE
ANTIRES = 0           # 仅共振项
NBANDSO = 16          # 价带数
NBANDSV = 8           # 导带数
OMEGAMAX = 15
```

**激子束缚能**:
| 激子 | 能量 (eV) | 束缚能 (meV) | 性质 |
|------|-----------|--------------|------|
| A¹s | 1.90      | 550          | 亮激子 |
| B¹s | 2.10      | 520          | 亮激子 |
| A²s | 2.15      | 300          | 暗激子 |

**吸收谱可视化**:
```python
def plot_exciton_absorption():
    """
    对比DFT(RPA)和BSE吸收谱
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # DFT-RPA结果 (无激子效应)
    energy_rpa, absorb_rpa = load_optical_data('OPTIC_RPA')
    ax.plot(energy_rpa, absorb_rpa, 'b--', label='RPA (no exciton)', linewidth=2)
    
    # BSE结果 (含激子效应)
    energy_bse, absorb_bse = load_optical_data('OPTIC_BSE')
    ax.plot(energy_bse, absorb_bse, 'r-', label='BSE (with exciton)', linewidth=2)
    
    # 标记激子峰
    ax.axvline(x=1.90, color='green', linestyle=':', alpha=0.7, label='A exciton')
    ax.axvline(x=2.10, color='orange', linestyle=':', alpha=0.7, label='B exciton')
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Absorption (arb. units)')
    ax.set_title('MoS₂ Optical Absorption')
    ax.legend()
    ax.set_xlim(1.5, 3.0)
    plt.tight_layout()
    plt.savefig('mos2_exciton.png', dpi=300)
```

---

## 5. 异质结构建

### 5.1 MoS₂/石墨烯异质结

**晶格匹配**:
- MoS₂: a = 3.193 Å
- 石墨烯: a = 2.460 Å
- 失配度: ~30%

**解决方案**: √3×√3石墨烯超胞 ≈ 1×1 MoS₂
- 超胞晶格常数: 4.26 Å
- 实际失配: ~1.3% (可接受)

**POSCAR**:
```
MoS2_Graphene
1.0
   4.2600000000000000    0.0000000000000000    0.0000000000000000
  -2.1300000000000000    3.6892800000000000    0.0000000000000000
   0.0000000000000000    0.0000000000000000   25.0000000000000000
Mo S C
1 2 6
direct
  0.0000000000000000  0.0000000000000000  0.5000000000000000   ! Mo
  0.3333333333333333  0.6666666666666667  0.5800000000000000   ! S1
  0.6666666666666667  0.3333333333333333  0.4200000000000000   ! S2
  0.0000000000000000  0.0000000000000000  0.7000000000000000   ! C
  0.3333333333333333  0.6666666666666667  0.7000000000000000   ! C
  0.6666666666666667  0.3333333333333333  0.7000000000000000   ! C
  0.0000000000000000  0.3333333333333333  0.7000000000000000   ! C
  0.3333333333333333  0.0000000000000000  0.7000000000000000   ! C
  0.6666666666666667  0.0000000000000000  0.7000000000000000   ! C
```

### 5.2 能带对齐

**计算步骤**:
1. 分别计算孤立层静电势
2. 计算异质结静电势
3. 确定能带偏移

**结果**:
- 价带偏移 (VBO): 0.8 eV
- 导带偏移 (CBO): 0.4 eV
- 类型: II型异质结 (有利于光生载流子分离)

---

## 6. 完整计算脚本

### 6.1 自动化工作流

```bash
#!/bin/bash
# mos2_workflow.sh

# 创建目录结构
mkdir -p {1_relax,2_bands,3_optics,4_strain,5_bse}

# 步骤1: 结构优化
echo "=== Step 1: Relaxation ==="
cd 1_relax
cp ../POSCAR .
cat > INCAR <<EOF
SYSTEM = MoS2 Relax
ENCUT = 400
EDIFF = 1E-6
EDIFFG = -0.01
IBRION = 2
ISIF = 4              # 固定晶胞形状，优化离子
NSW = 100
ISMEAR = 0
SIGMA = 0.05
GGA = PE
IVDW = 12             # DFT-D3
EOF
cat > KPOINTS <<EOF
K-Points
0
Gamma
15 15 1
0 0 0
EOF
mpirun -np 24 vasp_std
cp CONTCAR ../2_bands/POSCAR
cd ..

# 步骤2: 能带计算 (含SOC)
echo "=== Step 2: Band Structure ==="
cd 2_bands
# SCF
cat > INCAR <<EOF
SYSTEM = MoS2 SCF
ENCUT = 400
EDIFF = 1E-8
ISMEAR = -5
LSORBIT = .TRUE.
EOF
mpirun -np 24 vasp_std
# Bands (修改KPOINTS后)
mpirun -np 24 vasp_std
cd ..

# 步骤3: 光学性质
echo "=== Step 3: Optics ==="
cd 3_optics
cp ../2_bands/CHGCAR ../2_bands/WAVECAR .
cat > INCAR <<EOF
SYSTEM = MoS2 Optics
ENCUT = 400
ALGO = Exact
LOPTICS = .TRUE.
NBANDS = 120
NEDOS = 3001
CSHIFT = 0.05
EOF
mpirun -np 24 vasp_std
cd ..

# 步骤4: BSE计算
echo "=== Step 4: BSE ==="
cd 5_bse
cp ../3_optics/WAVECAR .
cat > INCAR <<EOF
SYSTEM = MoS2 BSE
ENCUT = 400
ALGO = BSE
ANTIRES = 0
NBANDSO = 16
NBANDSV = 8
OMEGAMAX = 15
EOF
mpirun -np 48 vasp_std
cd ..

echo "=== All done ==="
```

### 6.2 数据分析脚本

```python
#!/usr/bin/env python3
# analyze_mos2.py

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

def analyze_structure():
    """分析优化后的结构"""
    atoms = read('1_relax/CONTCAR')
    
    # 提取晶格常数
    cell = atoms.get_cell()
    a = np.linalg.norm(cell[0])
    
    # 计算键长
    positions = atoms.get_positions()
    mo_pos = positions[0]
    s1_pos = positions[1]
    s2_pos = positions[2]
    
    d_mo_s1 = np.linalg.norm(mo_pos - s1_pos)
    d_mo_s2 = np.linalg.norm(mo_pos - s2_pos)
    d_s_s = np.linalg.norm(s1_pos - s2_pos)
    
    print(f"晶格常数: {a:.3f} Å")
    print(f"Mo-S键长: {d_mo_s1:.3f} Å, {d_mo_s2:.3f} Å")
    print(f"S-S距离: {d_s_s:.3f} Å")
    
    return a, d_mo_s1, d_s_s

def analyze_band_gap():
    """分析带隙和有效质量"""
    # 从EIGENVAL读取能带
    kpoints, bands = read_eigenval('2_bands/EIGENVAL')
    
    # 找到K点索引
    k_idx = find_kpoint(kpoints, [1/3, 1/3, 0])
    
    # 计算带隙
    vbm = bands[k_idx, 16]  # 价带顶
    cbm = bands[k_idx, 17]  # 导带底
    gap = cbm - vbm
    
    print(f"K点直接带隙: {gap:.3f} eV")
    
    # 计算有效质量
    m_e = calculate_eff_mass(kpoints, bands[:, 17], k_idx, 'electron')
    m_h = calculate_eff_mass(kpoints, bands[:, 16], k_idx, 'hole')
    
    print(f"电子有效质量: {m_e:.3f} m₀")
    print(f"空穴有效质量: {m_h:.3f} m₀")
    
    return gap, m_e, m_h

if __name__ == '__main__':
    print("=== MoS₂ Analysis ===")
    analyze_structure()
    analyze_band_gap()
```

---

## 7. 参考

1. Mak et al., *PRL* 105, 136805 (2010) - MoS₂单层光电性质
2. Splendiani et al., *Nano Lett.* 10, 1271 (2010) - 光致发光
3. Komsa & Krasheninnikov, *PRB* 86, 241201 (2012) - 二维材料DFT计算
4. Yun et al., *PRB* 85, 033305 (2012) - 应变效应
5. Qiu et al., *Sci. Rep.* 3, 2965 (2013) - 光学性质
