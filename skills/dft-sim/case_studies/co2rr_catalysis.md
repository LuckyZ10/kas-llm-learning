# CO₂还原反应 (CO2RR) 电催化剂设计

## 背景

CO₂电化学还原(CO2RR)是应对气候变化的战略性技术。DFT计算用于筛选催化剂、预测活性和选择性，理解反应机理。

---

## 反应机理与路径

### 主要产物路径

| 产物 | 电子数 | 关键中间体 | 理论过电位 |
|------|--------|-----------|-----------|
| CO | 2e⁻ | *COOH, *CO | ~0.4 V |
| HCOOH | 2e⁻ | *OCHO, *HCOOH | ~0.6 V |
 | CH₃OH | 6e⁻ | *CO, *CHO, *CH₃O | ~0.8 V |
 | CH₄ | 8e⁻ | *CO, *CHO, *CH₃ | ~0.6 V |
| C₂H₄ | 12e⁻ | *CO-CO, *OCCO | ~0.4 V |

### 质子-电子耦合转移 (PCET)

```python
co2rr_mechanism = """
CO2RR关键基元步骤:

1. CO2活化 → *COOH (或 *OCHO)
   ΔG = G(*COOH) - G(CO2) - G(H+) - G(e-)
   
2. 分支点: 
   *COOH → *CO (放CO)
   *OCHO → HCOOH
   
3. CO深度还原 (若追求烃类):
   *CO + H+ + e- → *CHO
   *CHO + H+ + e- → *CH2O
   ... → CH4/CH3OH

4. C-C偶联 (C2+产物):
   2*CO → *CO-CO → *OCCO
   ... → C2H4/C2H5OH

DFT计算方法:
- 计算每个中间体吸附能
- 使用CHE模型 (Computational Hydrogen Electrode)
- 确定决速步 (limiting potential)
"""
```

---

## 计算方法

### 1. CHE模型实现

```python
#!/usr/bin/env python3
"""CO2RR自由能计算 - CHE模型"""

import numpy as np

class CHEModel:
    """Computational Hydrogen Electrode Model"""
    
    def __init__(self, temperature=298.15, ph=0):
        self.T = temperature
        self.pH = ph
        self.kT = 8.617e-5 * temperature  # eV
        
    def free_energy_correction(self, e_dft, zpe, s, temp=None):
        """计算自由能 G = E_DFT + ZPE - TS"""
        if temp is None:
            temp = self.T
        
        G = e_dft + zpe - temp * s / 96485  # eV (S in J/mol/K)
        return G
    
    def proton_electron_free_energy(self, u, ph=None):
        """计算1/2 H2 → H+ + e- 的自由能
        
        Args:
            u: 电极电位 (vs RHE)
            ph: pH值
        """
        if ph is None:
            ph = self.pH
            
        # CHE参考: H+ + e- ↔ 1/2 H2
        # 在U_RHE = 0, pH = 0时 ΔG = 0
        # 一般表达式:
        delta_g = -u - 0.059 * ph  # eV
        
        return delta_g
    
    def co2rr_step_energy(self, g_initial, g_final, u, n_proton=1):
        """计算CO2RR单步自由能变化"""
        # ΔG = G_final - G_initial - n*(G_H+ + G_e-)
        g_proton_electron = self.proton_electron_free_energy(u)
        delta_g = g_final - g_initial - n_proton * g_proton_electron
        
        return delta_g
    
    def limiting_potential(self, steps_energies):
        """计算极限电位
        
        UL = -max(ΔG) for all steps
        """
        max_delta_g = max(steps_energies)
        u_L = -max_delta_g
        
        return u_L

# 示例: Cu(111)上CO2→CO
def calculate_co2_to_co():
    """计算CO2还原为CO的自由能图"""
    
    che = CHEModel(temperature=298.15, ph=0)
    
    # 典型DFT能量 (已含ZPE校正, 单位eV)
    # 参考: K. Chan et al., JACS 2019
    energies = {
        'CO2_gas': 0.0,      # 参考态
        '*COOH': 0.55,       # CO2 + H+ + e- → *COOH
        '*CO': -0.15,        # *COOH + H+ + e- → *CO + H2O
        'CO_gas': -0.53,     # *CO → CO(g) + *
    }
    
    # 构建反应路径
    U_range = np.linspace(-1.0, 0.5, 100)
    
    print("="*60)
    print("CO2RR on Cu(111): CO2 → CO")
    print("="*60)
    
    # 在U = 0 V vs RHE
    u = 0.0
    step1 = che.co2rr_step_energy(energies['CO2_gas'], 
                                   energies['*COOH'], u, n_proton=1)
    step2 = che.co2rr_step_energy(energies['*COOH'],
                                   energies['*CO'], u, n_proton=1)
    step3 = energies['CO_gas'] - energies['*CO']  # 脱附
    
    print(f"Step 1: CO2 → *COOH    ΔG = {step1:+.2f} eV")
    print(f"Step 2: *COOH → *CO    ΔG = {step2:+.2f} eV")
    print(f"Step 3: *CO → CO(g)    ΔG = {step3:+.2f} eV")
    
    u_L = che.limiting_potential([step1, step2, step3])
    print(f"\nLimiting potential: U_L = {u_L:.2f} V vs RHE")
    print(f"Overpotential: η = {(-0.11 - u_L):.2f} V (vs thermo -0.11V)")
    
    return u_L

if __name__ == '__main__':
    calculate_co2_to_co()
```

### 2. VASP计算设置

```bash
# CO2RR吸附能计算示例
# INCAR
SYSTEM = CO2RR on Cu(111)
ENCUT = 500
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-6

# 表面模型
NSW = 0          # 单点计算或
IBRION = 2       # 吸附结构优化
ISIF = 0         # 优化离子位置
EDIFFG = -0.03   # 力收敛标准

# 偶极修正 (重要!)
LDIPOL = .TRUE.
IDIPOL = 3       # z方向
DIPOL = 0.5 0.5 0.5

# 溶剂化效应 (VASPsol)
LSOL = .TRUE.
EB_K = 80.0      # 介电常数 (水)
TAU = 0.0        # 表面张力

# 输出控制
LCHARG = .TRUE.
LWAVE = .FALSE.
LVHAR = .TRUE.   # 输出Hartree势 (用于分析)
```

```bash
# KPOINTS
k-Points
0
Gamma
4 4 1
0 0 0
```

```bash
# POSCAR - Cu(111) + *COOH
Cu_surface_CO2RR
   1.00000000000000
     5.108229160   0.000000000   0.000000000
     2.554114580   4.423849932   0.000000000
     0.000000000   0.000000000  30.000000000
   Cu  C  O  H
   27   1   2   1
Cartesian
  0.000000000   0.000000000   0.000000000  # Cu layer 1
  2.554114580   1.474616644   0.000000000
  # ... 更多Cu原子
  1.277057290   2.211924966   2.087000000  # C of *COOH
  0.877057290   1.611924966   3.287000000  # O1
  2.077057290   2.811924966   1.587000000  # O2
  0.377057290   1.111924966   3.787000000  # H
```

### 3. 吸附结构构建

```python
#!/usr/bin/env python3
"""构建CO2RR中间体吸附结构"""

from ase import Atoms
from ase.build import surface, molecule, add_adsorbate
from ase.io import write

def build_cooh_on_cu111():
    """构建Cu(111)上的*COOH结构"""
    
    # Cu(111)表面
    Cu_bulk = Atoms('Cu', cell=[3.61, 3.61, 3.61], pbc=True)
    Cu_slab = surface(Cu_bulk, (1,1,1), layers=4)
    Cu_slab.center(vacuum=15, axis=2)
    
    # *COOH几何 (弯曲构型)
    # C连接表面，COOH弯曲
    cooh = Atoms('COOH',
                 positions=[[0, 0, 0],      # C
                           [1.2, 0, 0.3],    # O1 (羰基)
                           [-0.8, 0.8, 0.8], # O2 (羟基)
                           [-1.2, 1.4, 0.2]]) # H
    
    # 添加到表面 (top位)
    add_adsorbate(Cu_slab, cooh, height=2.0, position=(0, 0))
    
    write('cooh_cu111.vasp', Cu_slab, format='vasp', direct=True)
    
    return Cu_slab

def build_all_intermediates():
    """生成CO2RR所有关键中间体"""
    
    intermediates = {
        '*COOH': {'formula': 'COOH', 'sites': ['top', 'bridge', 'hcp', 'fcc']},
        '*CO': {'formula': 'CO', 'sites': ['top', 'hollow']},
        '*CHO': {'formula': 'CHO', 'sites': ['top', 'bridge']},
        '*CH2O': {'formula': 'CH2O', 'sites': ['top']},
        '*OCHO': {'formula': 'OCHO', 'sites': ['bridge']},
        '*OCCO': {'formula': 'C2O2', 'sites': ['bridge']},
    }
    
    for name, info in intermediates.items():
        for site in info['sites']:
            # 构建并优化每个结构
            # 保存为POSCAR_{name}_{site}
            pass

if __name__ == '__main__':
    slab = build_cooh_on_cu111()
    print(f"Created slab with {len(slab)} atoms")
```

---

## 催化剂筛选

### 火山曲线分析

```python
#!/usr/bin/env python3
"""CO2RR火山曲线分析"""

import numpy as np
import matplotlib.pyplot as plt

def scaling_relations():
    """中间体吸附能标度关系"""
    
    # 典型标度关系 (Nørskov et al.)
    # ΔG(*COOH) ≈ 0.8 * ΔG(*CO) + 1.5
    # ΔG(*CHO) ≈ 1.0 * ΔG(*CO) + 0.5
    
    dG_CO = np.linspace(-2.5, 0.5, 100)
    dG_COOH = 0.8 * dG_CO + 1.5
    dG_CHO = 1.0 * dG_CO + 0.5
    
    # 理论过电位
    # CO路径: max(ΔG_COOH, ΔG_CO - ΔG_COOH)
    # 最优点在 ΔG_COOH ≈ ΔG_CO - ΔG_COOH → ΔG_CO ≈ 2*ΔG_COOH
    
    return dG_CO, dG_COOH, dG_CHO

def volcano_plot_co2rr():
    """绘制CO2RR火山曲线"""
    
    # 文献数据 (简化示例)
    catalysts = {
        'Au(111)': {'x': 0.3, 'eta': 0.45},
        'Ag(111)': {'x': 0.1, 'eta': 0.55},
        'Cu(111)': {'x': -0.15, 'eta': 0.35},
        'Cu(211)': {'x': -0.25, 'eta': 0.30},
        'Cu(100)': {'x': -0.30, 'eta': 0.25},
        'Ni(111)': {'x': -0.8, 'eta': 0.60},
        'Pt(111)': {'x': -1.2, 'eta': 0.80},  # HER主导
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, data in catalysts.items():
        ax.scatter(data['x'], data['eta'], s=100, label=name)
    
    # 理论火山曲线
    x_theory = np.linspace(-2.0, 1.0, 100)
    # 简化模型
    y_theory = np.maximum(0.5 + x_theory, 0.5 - x_theory)
    ax.plot(x_theory, y_theory, 'k--', label='Theory')
    
    ax.set_xlabel('ΔG(*CO) (eV)', fontsize=12)
    ax.set_ylabel('Overpotential η (V)', fontsize=12)
    ax.set_title('CO2RR Volcano Plot (CO pathway)', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('co2rr_volcano.png', dpi=150)
    
    return fig

if __name__ == '__main__':
    volcano_plot_co2rr()
```

### 单原子催化剂

```python
sac_co2rr = """
单原子催化剂 (SAC) CO2RR研究:

典型体系:
1. M-N-C (M = Fe, Co, Ni, Cu)
   - M-N4配位
   - 吡啶/吡咯N
   
2. 计算要点:
   - 自旋态 (高自旋/低自旋)
   - U值 (DFT+U)
   - 分散力 (DFT-D3)
   
3. 活性描述符:
   - d带中心
   - M-N键长
   - 电荷转移

4. 优势:
   - 原子利用率100%
   - 独特电子结构
   - 可调活性位点
"""
```

---

## 结果分析

### 自由能图绘制

```python
def plot_free_energy_diagram(potential=0.0):
    """绘制CO2RR自由能图"""
    
    # 反应路径和能量
    steps = ['CO₂ + *', '*COOH', '*CO', 'CO + *']
    energies_0V = [0.0, 0.55, -0.15, -0.53]  # vs CO2(g)
    
    # 在电位U下的能量
    # 每个PCET步的能量随U变化
    energies_U = [0.0,
                  0.55 - (-1)*potential,  # CO2→*COOH: 1e
                  -0.15 - (-1)*potential, # *COOH→*CO: 1e  
                  -0.53]                   # 脱附: 无e
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 阶梯图
    x = np.arange(len(steps))
    for i in range(len(steps)-1):
        ax.plot([x[i], x[i+1]], [energies_U[i], energies_U[i]], 'b-', lw=2)
        ax.plot([x[i+1], x[i+1]], [energies_U[i], energies_U[i+1]], 'b-', lw=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=15, ha='right')
    ax.set_ylabel('Free Energy (eV)', fontsize=12)
    ax.set_title(f'CO2RR Free Energy Diagram @ U = {potential} V', fontsize=14)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 标出决速步
    max_step = np.argmax(np.diff(energies_U))
    ax.annotate('RDS', xy=(max_step+0.5, (energies_U[max_step]+energies_U[max_step+1])/2),
                fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig(f'free_energy_{potential}V.png', dpi=150)

if __name__ == '__main__':
    plot_free_energy_diagram(potential=0.0)
    plot_free_energy_diagram(potential=-0.5)
```

### 选择性与活性的竞争

```python
def analyze_selectivity():
    """分析HER vs CO2RR选择性"""
    
    # HER决速步: H2O → *H + H+ + e-
    # CO2RR决速步: CO2 → *COOH
    
    catalysts_data = {
        'Cu(111)': {'G_H': 0.15, 'G_COOH': 0.55},
        'Cu(211)': {'G_H': 0.10, 'G_COOH': 0.45},
        'Au(111)': {'G_H': 0.35, 'G_COOH': 0.65},
        'Ag(111)': {'G_H': 0.55, 'G_COOH': 0.75},
    }
    
    print("="*60)
    print("Selectivity Analysis: CO2RR vs HER")
    print("="*60)
    print(f"{'Surface':<15}{'ΔG(*H)':<12}{'ΔG(*COOH)':<15}{'Selectivity':<15}")
    print("-"*60)
    
    for name, data in catalysts_data.items():
        g_h = data['G_H']
        g_cooh = data['G_COOH']
        
        # 选择性判据: 更倾向于CO2RR如果 G_COOH < G_H
        if g_cooh < g_h:
            selectivity = "CO2RR favored"
        else:
            selectivity = "HER favored"
            
        print(f"{name:<15}{g_h:<12.2f}{g_cooh:<15.2f}{selectivity:<15}")
```

---

## 先进方法

### 显式溶剂化

```bash
# 显式水层 + VASPsol
# 在slab上添加水层

# 使用packmol构建初始构型
cat > water_packmol.in << 'EOF'
tolerance 2.0
filetype xyz
output water_layer.xyz

structure water.xyz
  number 64
  inside box 0. 0. 10. 10.2 10.2 20.
end structure
EOF

packmol < water_packmol.in
```

### 动力学效应

```python
aimd_co2rr = """
CO2RR动力学研究 (AIMD):

1. 预平衡
   - NVT系综, 300K
   - 1-2 ps平衡
   
2. 反应模拟
   - 约束MD (metadynamics)
   - 自由能计算
   
3. 溶剂重组
   - 电场响应
   - 氢键网络重排

4. 软件
   - CP2K: CPMD/MM
   - VASP: AIMD
   - Quantum ESPRESSO: CP
"""
```

---

## 参考资源

- 综述: Nitopi et al., "Progress and Perspectives of Electrochemical CO2 Reduction", Chem. Rev. 2019
- CHE模型: Nørskov et al., J. Phys. Chem. B 2004
- VASPsol: Mathew et al., J. Chem. Phys. 2014
- 数据库: CatApp (https://catapp.stanford.edu/)

---

*案例作者: DFT-Sim Team*
*最后更新: 2026-03-08*
