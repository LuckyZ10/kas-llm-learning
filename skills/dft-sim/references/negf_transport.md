# NEGF 非平衡格林函数方法

## 简介

非平衡格林函数 (Non-Equilibrium Green's Function, NEGF) 方法是研究量子输运的标准理论框架，用于计算纳米器件的电子输运性质、I-V特性、量子电导等。

---

## 理论基础

### 开放系统的格林函数

对于分为左电极(L)、中心散射区(C)、右电极(R)的体系：

$$\hat{H} = \begin{pmatrix} H_L & V_{LC} & 0 \\ V_{CL} & H_C & V_{CR} \\ 0 & V_{RC} & H_R \end{pmatrix}$$

中心区的推迟格林函数：
$$G^r(E) = \left[ (E+i\eta)S_C - H_C - \Sigma_L(E) - \Sigma_R(E) \right]^{-1}$$

其中 $\Sigma_{L/R}$ 是自能，描述电极对中心区的耦合。

### 输运方程

**Landauer-Büttiker公式**:
$$I = \frac{2e}{h} \int_{-\infty}^{\infty} T(E) \left[ f_L(E) - f_R(E) \right] dE$$

**透射系数**:
$$T(E) = \text{Tr}[\Gamma_L G^r \Gamma_R G^a]$$

其中 $\Gamma_{L/R} = i(\Sigma^r - \Sigma^a)$ 是线宽函数。

### 自洽NEGF-DFT

电荷密度自洽计算：
$$\rho = -\frac{i}{2\pi} \int_{-\infty}^{\infty} G^<(E) dE$$

其中小于格林函数:
$$G^< = G^r \left( f_L \Gamma_L + f_R \Gamma_R \right) G^a$$

---

## 软件实现

### 1. TranSIESTA / Smeagol

TranSIESTA是基于SIESTA的NEGF-DFT实现。

```bash
# 1. 构建电极 (重复单元)
cat > electrode.fdf << 'EOF'
SystemName          Au electrode
NumberOfAtoms       9
NumberOfSpecies     1

%block ChemicalSpeciesLabel
1 79 Au
%endblock

LatticeConstant     1.0 Ang
%block LatticeVectors
  8.16  0.00  0.00
  0.00  8.16  0.00
  0.00  0.00  4.08
%endblock

%block AtomicCoordinatesAndAtomicSpecies
  0.000  0.000  0.000  1
  2.040  2.040  0.000  1
  # ... 其他原子
%endblock

# 电极设置
ElectrodeLeft
  used-geometry     electrode.TSHS
  used-atoms        1..9
  used-cell         0 0 1
EndElectrodeLeft

SolutionMethod      TRANSIESTA
TS.Voltage          0.0 eV    # 偏置电压
TS.Elecs.Bulk       true      # 体电极近似
TS.Elecs.DM.Update  cross-terms
EOF

# 2. 运行电极计算
siesta < electrode.fdf > electrode.out

# 3. 构建散射区 (电极+分子+电极)
cat > scattering.fdf << 'EOF'
SystemName          Au-Molecule-Au junction

# 包含电极信息
%include electrode.TSHS

%block AtomicCoordinatesAndAtomicSpecies
# 左电极 (27 atoms)
# 分子 (如苯二硫醇, 6C+2S+4H)
# 右电极 (27 atoms)
%endblock

# NEGF-DFT设置
SolutionMethod      TRANSIESTA
TS.Voltage          0.0 eV    # 从0开始

# 能量网格
TS.Elecs.Eta        0.0001 Ry
TS.Contours.Eq.Pt   20
TS.Contours.NEq.Pt  20

# 自洽设置
DM.MixingWeight     0.1
DM.NumberPulay      5

# I-V扫描
%block TS.Voltage.Contour
 0.0  1.0  0.2    # 起始 终止 步长 (V)
%endblock
EOF

# 4. 运行输运计算
siesta < scattering.fdf > scattering.out

# 5. 分析输运性质
tbtrans < tbtrans.fdf > tbtrans.out
```

```bash
# tbtrans后处理输入
cat > tbtrans.fdf << 'EOF'
SystemLabel         scattering
TSHSFile            scattering.TSHS

# 电压点
Voltage             0.0 eV

# 能量网格
%block TBT.Energy.Grid
  -5.0 eV   5.0 eV   0.01 eV  # Emin Emax dE
%endblock

# 透射分析
TBT.T.Eig           10          # 前10个本征通道
TBT.DOS.Gf          true        # 局域DOS
TBT.Current.Orb     true        # 轨道电流

# 投影DOS
%block TBT.Proj
  1..6          # 分子区域
%endblock
EOF
```

### 2. QuantumATK (Synopsys)

商业软件，提供图形界面和Python API。

```python
#!/usr/bin/env python3
"""QuantumATK NEGF-DFT输运计算示例"""

# 注意: 需QuantumATK Python环境

from atk import *

# 构建金电极
left_electrode = Electrode(
    bulk_configuration=BulkConfiguration(
        brillouin_zone_setting=MonkhorstPackGrid(1, 1, 100),
        elements=Gold,
        lattice=FaceCenteredCubic(4.08*Angstrom)
    )
)

# 构建分子结
molecule = MoleculeConfiguration(
    elements=[Carbon, Carbon, Sulfur, Sulfur],
    # 苯二硫醇坐标...
)

# 构建中心散射区
central_region = CentralRegion(
    left_electrode=left_electrode,
    right_electrode=left_electrode,  # 对称电极
    molecule=molecule,
    electrode_spacing=5.0*Angstrom
)

# NEGF-DFT计算
calculator = DeviceLCAOCalculator(
    exchange_correlation=PBE.PBE,
    basis_set=Gold_DZP,
    k_point_sampling=MonkhorstPackGrid(1, 1, 100),
    device_algorithm=NEGF(
        electrode_voltages=[0.0*Volt, 0.0*Volt],  # 初始无偏压
        energy_zero=AverageFermiLevel,
        contour_integration_point_density=1000
    )
)

central_region.setCalculator(calculator)
central_region.update()

# 计算零偏压透射谱
transmission = TransmissionSpectrum(
    configuration=central_region,
    energies=numpy.linspace(-3, 3, 1000)*eV,
    self_energy_calculator=RecursionSelfEnergy()
)

# I-V特性计算
iv_curve = IVCurve(
    configuration=central_region,
    voltage_range=[0.0, 1.0, 0.1]*Volt,  # 0-1V, 步长0.1V
    temperature=300*Kelvin
)
```

### 3. SMEAGOL (基于SIESTA)

```bash
# SMEAGOL电极计算
cat > elec.fdf << 'EOF'
SystemName          Au Electrode

NumberOfAtoms       9
NumberOfSpecies     1

%block ChemicalSpeciesLabel
 1  79  Au
%endblock

LatticeConstant 1.0 Ang
%block LatticeVectors
 8.16  0.00  0.00
 0.00  8.16  0.00
 0.00  0.00  4.08
%endblock

SolutionMethod      Diagon
ElectronicTemperature 300 K

%block kgrid_Monkhorst_Pack
  3   0   0   0.0
  0   3   0   0.0
  0   0  20   0.0
%endblock
EOF

# SMEAGOL输运计算
cat > smeagol.fdf << 'EOF'
SystemName          Au-C60-Au Junction

# 电极设置
MD.LeftElecFile     elec.TSHS
MD.LeftElecShift    0.0
MD.RightElecFile    elec.TSHS
MD.RightElecShift   0.0

# 电压循环
MD.VoltageMin       0.0
MD.VoltageMax       2.0
MD.VoltageStep      0.1

# NEGF设置
MD.Niter            100
MD.Tol              1.0d-4
MD.eta              1.0d-4

# 能量积分
MD.Emin            -10.0
MD.Emax             10.0
MD.NumEnergies      1000

SolutionMethod      NEGF
EOF

# 运行
smeagol < smeagol.fdf > smeagol.out
```

### 4. GOLLUM (紧束缚/有效质量)

适用于大尺度量子输运。

```fortran
! GOLLUM输入示例
TypeOfLead        supercell
TypeOfSystem      normal

! 哈密顿量文件
BulkLead1         lead1_hamiltonian.dat
BulkLead2         lead2_hamiltonian.dat
ScatteringRegion  device_hamiltonian.dat

! 能量范围
EMin             -2.0
EMax              2.0
NEnergies         1000

! 温度
Temperature       300.0

! 输出选项
Output            transmission
Output            dos
Output            current
```

---

## 输运分析

### 1. 透射谱解析

```python
#!/usr/bin/env python3
"""解析NEGF输运计算结果"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def parse_transmission(file_path):
    """解析透射系数文件"""
    data = np.loadtxt(file_path, skiprows=1)
    energy = data[:, 0]      # eV
    transmission = data[:, 1]  # T(E)
    return energy, transmission

def plot_transmission(energy, transmission, fermi_level=0.0, 
                       dos=None, labels=None):
    """绘制透射谱与DOS"""
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # 透射谱
    ax1 = axes[0]
    ax1.plot(energy, transmission, 'b-', lw=2, label='T(E)')
    ax1.axvline(x=fermi_level, color='r', linestyle='--', label='$E_F$')
    ax1.set_ylabel('Transmission T(E)', fontsize=12)
    ax1.set_title('NEGF Transmission Spectrum', fontsize=14)
    ax1.legend()
    ax1.set_ylim(0, min(max(transmission)*1.1, 5))
    
    # DOS (可选)
    if dos is not None:
        ax2 = axes[1]
        ax2.plot(energy, dos, 'g-', lw=2, label='DOS')
        ax2.axvline(x=fermi_level, color='r', linestyle='--')
        ax2.set_xlabel('Energy (eV)', fontsize=12)
        ax2.set_ylabel('DOS (states/eV)', fontsize=12)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('transmission.png', dpi=150)
    return fig

def calculate_conductance(energy, transmission, fermi_level=0.0, temp=300):
    """计算电导 (G0 = 2e²/h)"""
    kB = 8.617e-5  # eV/K
    beta = 1.0 / (kB * temp)
    
    # 量子电导 (零温)
    G0 = transmission[np.argmin(np.abs(energy - fermi_level))]
    
    # 有限温度电导 (Landauer公式积分)
    def fermi_derivative(E, mu):
        """Fermi函数导数 -df/dE"""
        x = beta * (E - mu)
        # 避免溢出
        x = np.clip(x, -100, 100)
        return beta * np.exp(x) / (1 + np.exp(x))**2
    
    integrand = transmission * fermi_derivative(energy, fermi_level)
    G_finite_T = integrate.simps(integrand, energy)
    
    print(f"Zero-temperature conductance: {G0:.4f} G₀")
    print(f"Conductance at {temp}K: {G_finite_T:.4f} G₀")
    
    return G0, G_finite_T

def calculate_iv_curve(voltages, transmission_func, temp=300):
    """计算I-V曲线"""
    kB = 8.617e-5  # eV/K
    e_charge = 1.0  # 电子电荷 (自然单位)
    
    currents = []
    
    for V in voltages:
        mu_L = -V/2  # 左电极化学势
        mu_R = V/2   # 右电极化学势
        
        # 能量网格
        E = np.linspace(-5, 5, 2000)
        T_E = transmission_func(E)
        
        # 电流积分 (Landauer公式)
        def fermi(E, mu):
            x = (E - mu) / (kB * temp)
            x = np.clip(x, -100, 100)
            return 1.0 / (1 + np.exp(x))
        
        integrand = T_E * (fermi(E, mu_L) - fermi(E, mu_R))
        I = integrate.simps(integrand, E)
        currents.append(I)
    
    return np.array(currents)

if __name__ == '__main__':
    # 示例: 单能级模型透射
    E = np.linspace(-3, 3, 1000)
    epsilon0 = 0.0
    gamma = 0.1
    T = 4 * gamma**2 / ((E - epsilon0)**2 + (2*gamma)**2)  # Breit-Wigner
    
    plot_transmission(E, T)
    G0, GT = calculate_conductance(E, T)
    
    # I-V曲线
    V_range = np.linspace(0, 2, 21)
    I = calculate_iv_curve(V_range, lambda e: np.interp(e, E, T))
    
    plt.figure()
    plt.plot(V_range, I * 38.7, 'b-o', lw=2)  # μA
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Current (μA)', fontsize=12)
    plt.title('I-V Characteristics', fontsize=14)
    plt.savefig('iv_curve.png', dpi=150)
```

### 2. 本征通道分析

```python
def analyze_eigenchannels(transmission_matrix, num_channels=5):
    """分析透射本征通道"""
    # 奇异值分解
    U, s, Vh = np.linalg.svd(transmission_matrix)
    
    print("="*60)
    print("Eigenchannel Analysis")
    print("="*60)
    print(f"{'Channel':<10}{'Transmission':<15}{'Contribution':<15}")
    print("-"*60)
    
    T_total = np.sum(s)
    for i in range(min(num_channels, len(s))):
        contrib = s[i] / T_total * 100
        print(f"{i+1:<10}{s[i]:<15.6f}{contrib:<15.2f}%")
    
    print(f"\nTotal transmission: {T_total:.6f}")
    
    return s, U, Vh
```

---

## 常见应用

### 1. 分子电子器件

```bash
# 烷硫醇分子结计算流程

# 1. 优化分子结构
# 2. 构建电极-分子-电极构型
# 3. NEGF-DFT自洽计算
# 4. 分析透射零点 (HOMO-LUMO间隙)

# 关键参数
echo "分子电子器件关键考虑:"
echo "1. 电极-分子界面耦合"
echo "2. 分子能级对齐 (HOMO-LUMO vs EF)"
echo "3. 门电压效应"
echo "4. 拉伸-压缩效应"
```

### 2. 石墨烯/2D材料器件

```python
# 石墨烯纳米带输运
graphene_junction = """
石墨烯纳米带器件设置:

1. 电极: 半无限石墨烯薄片 (扶手椅或锯齿)
2. 中心区: 纳米带+缺陷/势垒
3. 边界条件: 周期性 (横向) + 开放 (输运方向)

透射特征:
- 完美纳米带: 量子化平台 (T = n × G₀)
- 缺陷散射: 透射抑制
- 边缘态: 导电通道
"""
```

### 3. 自旋输运

```bash
# 自旋极化NEGF-DFT (QuantumATK)

# 磁性电极 (如Co/MgO/Fe MTJ)
calculator = DeviceLCAOCalculator(
    spin_polarization=CollinearSpin,
    initial_spin=[[0,0,1], [0,0,-1]],  # 平行/反平行
    # ...
)

# 磁阻计算
TMR = (R_AP - R_P) / R_P * 100%  # 隧穿磁阻
```

### 4. 热输运 (声子NEGF)

```python
phonon_negf = """
声子NEGF方法 (ALAMODE, ShengBTE):

热导公式:
κ = (ħ²/2πkBT²) ∫ dω ω² T_ph(ω) n(ω)(n(ω)+1)

其中:
- T_ph(ω): 声子透射
- n(ω): Bose-Einstein分布

软件:
- ALAMODE: 基于DFT力常数
- ShengBTE: 体材料BTE
- GPUMD: 机器学习势+NEGF
"""
```

---

## 最佳实践

### 模型构建指南

```python
modeling_guide = {
    "电极": {
        "尺寸": "≥3×3原子面，避免边缘效应",
        "k点": "输运方向: 1点, 横向: 密集",
        "收敛": "体材料性质匹配",
    },
    "中心区": {
        "长度": "包含界面区域 + 缓冲层",
        "真空": "周期体系需足够真空 (>15Å)",
        "对称性": "尽可能利用减少计算量",
    },
    "耦合": {
        "距离": "自然键长或实验值",
        "测试": "透射对距离敏感性",
    }
}
```

### 收敛性测试

```bash
# 1. 电极尺寸收敛
echo "测试不同电极截面..."
for nx in 2 3 4 5; do
    # 运行计算
    echo "Electrode size ${nx}x${nx}: T(EF) = X.XX"
done

# 2. k点收敛
echo "测试k点密度..."
for nk in 10 20 40 80; do
    echo "k-points: $nk, T(EF) = X.XX"
done

# 3. 能量网格收敛
echo "测试能量网格..."
for ne in 500 1000 2000; do
    echo "Energy points: $ne, Current = X.XX"
done
```

### 常见问题

**问题1: SCF不收敛**
- 原因: 初始猜测差或电压过大
- 解决: 从0V开始，逐步增加电压
- 解决: 使用Pulay混合，降低mixing参数

**问题2: 负微分电阻 (NDR)**
- 可能物理效应
- 也可能是数值假象 (检查网格密度)

**问题3: 透射>1**
- 这是正常的 (多通道贡献)
- 单个通道T ≤ 1

---

## 参考资源

- TranSIESTA: https://siesta-project.org/siesta
- QuantumATK: https://www.synopsys.com/quantumatk
- SMEAGOL: https://www.smeagol.tcd.ie/
- GOLLUM: https://quantum-transport.org/gollum/
- 教材: Datta, "Electronic Transport in Mesoscopic Systems"
- 教材: Di Ventra, "Electrical Transport in Nanoscale Systems"

---

*文档版本: 1.0*
*最后更新: 2026-03-08*
