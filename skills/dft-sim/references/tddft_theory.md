# TDDFT 含时密度泛函理论

## 简介

含时密度泛函理论 (Time-Dependent DFT, TDDFT) 是研究激发态电子结构、光学吸收和电子动力学的标准方法，可计算激发能、振子强度和激发态几何结构。

---

## 理论基础

### 线性响应TDDFT

含时Kohn-Sham方程通过密度响应函数描述激发态：

$$\chi(r,r',\omega) = \chi_{KS}(r,r',\omega) + \int d r_1 d r_2 \chi_{KS}(r,r_1,\omega) \left[ \frac{1}{|r_1-r_2|} + f_{xc}(r_1,r_2,\omega) \right] \chi(r_2,r',\omega)$$

激发能通过Casida方程求解：
$$\begin{pmatrix} A & B \\ B^* & A^* \end{pmatrix} \begin{pmatrix} X \\ Y \end{pmatrix} = \omega \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \begin{pmatrix} X \\ Y \end{pmatrix}$$

### Tamm-Dancoff近似 (TDA)

忽略耦合矩阵B (即设Y=0)，得到简化方程：
$$A X = \omega X$$

TDA适用于:
- 单参考主导的激发态
- 大体系计算加速
- 避免电荷转移态的虚假根问题

---

## VASP实现

### 1. ALDA/TDLDA计算 (标准TDDFT)

```bash
# 步骤1: 基态计算
# INCAR.gs
SYSTEM = Ground State Calculation
ENCUT = 500
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-8
NSW = 0
LWAVE = .TRUE.
```

```bash
# 步骤2: TDDFT计算
# INCAR.tddft
SYSTEM = TDDFT Excitation
ENCUT = 500
ISMEAR = 0
SIGMA = 0.05

# TDDFT设置
ALGO = ALDA        # 绝热LDA (或BSE, TDHF)
NBANDS = 48        # 包含空带 (NVB+NCB)
NBANDSO = 16       # 占据带数
NBANDSV = 32       # 空带数
NOMEGA = 200       # 频率点数

# 可选: Tamm-Dancoff近似
LTAMMD = .TRUE.    # 开启TDA

# 可选: 激子计算 (BSE)
ALGO = BSE         # Bethe-Salpeter方程
```

```bash
# KPOINTS (单点用于分子/团簇)
k-Points
0
Gamma
1 1 1
0 0 0
```

### 2. TDDFT输出解析

```python
#!/usr/bin/env python3
"""解析VASP TDDFT/BSE结果"""

import numpy as np
import matplotlib.pyplot as plt

def parse_excitation_energies(outcar='OUTCAR', vasprun='vasprun.xml'):
    """解析激发能 """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(vasprun)
        root = tree.getroot()
        
        excitations = []
        for exc in root.findall(".//excitation"):
            energy = float(exc.find('energy').text)
            strength = float(exc.find('oscillator').text)
            excitations.append((energy, strength))
        
        return np.array(excitations)
    except:
        # 从OUTCAR解析
        with open(outcar, 'r') as f:
            lines = f.readlines()
        
        excitations = []
        for i, line in enumerate(lines):
            if 'Excitation energy' in line:
                energy = float(line.split()[-2])  # eV
                # 找下一行的振子强度
                if i+1 < len(lines) and 'oscillator strength' in lines[i+1]:
                    strength = float(lines[i+1].split()[-1])
                    excitations.append((energy, strength))
        
        return np.array(excitations)

def plot_absorption_spectrum(excitations, broadening=0.1, energy_range=None):
    """绘制吸收谱 (高斯展宽) """
    if len(excitations) == 0:
        print("No excitation data found")
        return
    
    energies = excitations[:, 0]
    strengths = excitations[:, 1]
    
    # 构建能量网格
    if energy_range is None:
        emin, emax = energies.min() - 1, energies.max() + 1
    else:
        emin, emax = energy_range
    
    e_grid = np.linspace(emin, emax, 1000)
    spectrum = np.zeros_like(e_grid)
    
    # 高斯展宽
    for e, s in zip(energies, strengths):
        spectrum += s * np.exp(-((e_grid - e)**2) / (2 * broadening**2))
    
    plt.figure(figsize=(8, 5))
    plt.plot(e_grid, spectrum, 'b-', lw=2, label=f'FWHM={broadening*2:.2f} eV')
    plt.vlines(energies, 0, strengths * spectrum.max() / strengths.max(), 
               colors='r', linestyles='--', alpha=0.5, label='Excitation energies')
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Intensity (arb. units)', fontsize=12)
    plt.title('TDDFT Absorption Spectrum', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('absorption_spectrum.png', dpi=150)
    print("Spectrum saved to absorption_spectrum.png")
    return e_grid, spectrum

if __name__ == '__main__':
    exc = parse_excitation_energies()
    print(f"Found {len(exc)} excitations")
    print("Energy(eV)  Oscillator Strength")
    for e, s in exc[:10]:  # 显示前10个
        print(f"{e:10.4f}  {s:12.6f}")
    
    plot_absorption_spectrum(exc, broadening=0.15)
```

---

## Quantum ESPRESSO实现 (TurboTDDFT)

### 1. 安装Turbo-Lanczos

```bash
# QE 7.x已包含turboTDDFT
./configure --with-turbo
make turbo-lanczos
```

### 2. 输入文件设置

```bash
# pw.x输入 (基态)
&CONTROL
calculation = 'scf',
prefix = 'molecule',
outdir = './tmp',
/
&SYSTEM
ibrav = 1,
celldm(1) = 20.0,
nat = 3,
ntyp = 2,
ecutwfc = 60,
ecutrho = 240,
/
&ELECTRONS
conv_thr = 1.0d-10,
/
ATOMIC_SPECIES
H 1.00794 H.pbe-van_bm.UPF
O 15.9994 O.pbe-van_bm.UPF
ATOMIC_POSITIONS angstrom
O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0
K_POINTS gamma
```

```bash
# turbo_davidson.x输入 (Davidson对角化)
&lr_dav
prefix = 'molecule',
outdir = './tmp',
evc1_filename = 'evc1.dat',
evc1_dir = './tmp',
eign = 10,              # 计算10个激发态
num_init = 40,          # 初始向量数
num_basis_max = 80,     # 最大基向量数
residue_conv_thr = 1.0d-4,
start = 'from_scratch',
ipol = 1,               # 偏振方向 (1=x, 2=y, 3=z, 4=平均)
/
```

```bash
# turbo_lanczos.x输入 (Lanczos迭代)
&lr_input
prefix = 'molecule',
outdir = './tmp',
eta = 0.005,            # 洛伦兹展宽
itermax = 500,          # Lanczos迭代数
ipol = 4,               # 各向同性
extrapolation = 'no',
e_psi = 0.05,
/
```

### 3. TurboTDDFT运行脚本

```bash
#!/bin/bash
# turbo_workflow.sh - TurboTDDFT完整流程

PREFIX="molecule"

# 1. 基态计算
mpirun pw.x < pw.in > pw.out

# 2. 准备波函数
mpirun turbo_lanczos.x < turbo_prep.in > turbo_prep.out

# 3. Lanczos迭代
mpirun turbo_lanczos.x < turbo_lanczos.in > turbo_lanczos.out

# 4. 后处理 (解析光谱)
mpirun turbo_spectrum.x < spectrum.in > spectrum.out

echo "TurboTDDFT calculation complete!"
```

---

## 实时TDDFT (rt-TDDFT)

### OCTOPUS实现 (VASP/GPAW/其他)

```bash
# octopus输入文件 inp
CalculationMode = td
ExperimentalFeatures = yes

# 系统设置
Dimensions = 3
Spacing = 0.2
Radius = 6.0

# 时间演化
TDTimeStep = 0.002
TDMaximumIter = 5000
TDPropagator = etrs

# 激光场
omega = 2.0*eV
tau0 = 5.0
 strength = 0.01
envelope = "sin^2"
%TDExternalFields
electric_field | 1 | 0 | 0 | omega | tau0 | strength | envelope
%

# 输出
TDOutput = density + multipoles
```

---

## 常见应用

### 1. 分子激发态计算

```python
#!/usr/bin/env python3
"""有机分子激发态系统计算 (TDDFT示例)"""

tddft_workflow = """
有机分子(如酞菁、并苯)激发态计算流程:

1. 基态优化
   - 几何优化至力<0.01 eV/Å
   - 注意: 泛函选择影响CT态描述
   
2. TDDFT设置
   - 常用泛函: B3LYP, CAM-B3LYP (改善CT态)
   - 长程校正: CAM-B3LYP, ωB97X-D
   
3. 溶剂效应
   VASP: 使用隐式溶剂模型
   QE: 使用Environ模块
   
4. 典型问题
   - CT态低估: 改用range-separated泛函
   - Rydberg态: 需扩散基函数
"""

# 推荐输入参数
tddft_params = {
    'functionals': {
        'B3LYP': '标准有机分子, 便宜',
        'CAM-B3LYP': '改善电荷转移态',
        'ωB97X-D': '含色散校正',
        'PBE0': '周期体系首选'
    },
    'basis': {
        'molecules': '增加弥散函数',
        'periodic': 'ENCUT≥1.3*ENMAX'
    },
    'convergence': {
        'NVB': '包含所有价带',
        'NCB': '至少20-30个导带',
        'NOMEGA': '200-400个频率点'
    }
}
```

### 2. 表面等离激元 (Plasmon)

```bash
# 金属纳米颗粒等离激元 (VASP TDDFT)
# INCAR
SYSTEM = Metal Nanoparticle Plasmon
ENCUT = 400
ISMEAR = 0
SIGMA = 0.05

# TDDFT for plasmon
ALGO = ALDA
NBANDS = 200        # 大体系需更多带
NBANDSO = 100
NBANDSV = 100
NOMEGA = 400
LCHARG = .FALSE.
LWAVE = .FALSE.
```

### 3. 光催化激发态动力学

```python
#!/usr/bin/env python3
"""光催化体系激发态分析"""

# 非绝热耦合计算 (NAMD)
namd_workflow = """
光催化NAMD计算流程:

1. 基态MD轨迹 (AIMD)
   - 1-10 ps轨迹
   - 保存每10fs的波函数
   
2. 面跳跃计算
   - 计算非绝热耦合(NAC)
   - 追踪电子-空穴复合
   
3. 常用软件
   - VASP: 结合PYXAID/NAMD
   - CP2K: 内置SH-NAMD
   - QChem: 内置FSSH
"""
```

---

## 结果分析

### 振子强度与跃迁偶极矩

```python
def analyze_transitions(excitations, coords=None):
    """分析跃迁性质
    
    Args:
        excitations: [(energy, strength, dipole), ...]
        coords: 原子坐标 (用于计算电荷转移)
    """
    print("="*60)
    print("TDDFT Transition Analysis")
    print("="*60)
    print(f"{'State':<8}{'Energy(eV)':<12}{'f':<10}{'Type':<20}")
    print("-"*60)
    
    for i, (e, f, *rest) in enumerate(excitations, 1):
        if f > 0.1:
            ttype = "Bright"
        elif f > 0.001:
            ttype = "Weak"
        else:
            ttype = "Dark"
        print(f"{i:<8}{e:<12.4f}{f:<10.6f}{ttype:<20}")
    
    # 激发能统计
    energies = [e for e, f in excitations]
    print(f"\nStatistical Summary:")
    print(f"  First excitation: {min(energies):.3f} eV")
    print(f"  Mean energy: {np.mean(energies):.3f} eV")
    print(f"  Strongest transition: {excitations[np.argmax([f for e,f in excitations])][0]:.3f} eV")
```

---

## 最佳实践

### 泛函选择指南

| 体系类型 | 推荐泛函 | 说明 |
|---------|---------|------|
| 小分子 | B3LYP, PBE0 | 平衡精度与效率 |
| CT态 | CAM-B3LYP, ωB97X | 长程校正必需 |
| 金属表面 | PBE, LDA | 简单泛函足够 |
| Rydberg态 | 含弥散基函数 | 需特殊处理 |

### 收敛性检查

```bash
# NBANDS收敛测试
for nb in 20 30 40 50 60; do
    cat > INCAR << EOF
ALGO = ALDA
NBANDSO = 16
NBANDSV = $nb
NOMEGA = 200
EOF
    mpirun vasp_std
cp vasprun.xml vasprun_nb${nb}.xml
done

# 比较第一激发能收敛
python -c "
import sys
for nb in [20,30,40,50,60]:
    # 解析激发能
    print(f'NBANDS={nb}: E1 = X.XX eV')
"
```

### 常见问题

**问题1: CT态能量严重低估**
- 解决: 使用CAM-B3LYP或ωB97X-D等range-separated泛函
- 解决: 增加空带数 (NBANDSV)

**问题2: 暗态(f≈0)计算**
- 解决: 这是物理正确的，无需担心
- 或: 检查对称性破缺

**问题3: 周期性体系激发**
- 注意: TDDFT仅适用于孤立体系或小超胞
- 大体系使用RPA或BSE

---

## 参考资源

- VASP Wiki: https://www.vasp.at/wiki/index.php/Time_dependent_DFT
- TurboTDDFT: http://www.tddft.org/programs/octopus/wiki/index.php/Turbo-TDDFT
- OCTOPUS: http://www.tddft.org/programs/octopus/
- Review: Ullrich, "Time-Dependent Density-Functional Theory"

---

*文档版本: 1.0*
*最后更新: 2026-03-08*
