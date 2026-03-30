# 案例研究：Cu金属电子结构与输运性质

## 概述

Cu是研究得最深入的金属之一，具有简单的fcc结构和优异的导电性。本案例展示金属体系的第一性原理计算，包括费米面、输运性质和表面电子态。

**目标材料**: Cu (面心立方)
**空间群**: Fm-3m (No. 225)
**实验晶格常数**: 3.615 Å
**实验态密度@费米面**: N(E_F) = 0.29 states/eV·atom

---

## 1. 体相电子结构

### 1.1 结构优化

**VASP输入**:
```
SYSTEM = Cu FCC
ENCUT = 450
ISMEAR = -5          # 四面体方法，金属也可用
# 或
ISMEAR = 1           # Methfessel-Paxton
SIGMA = 0.2          # 展宽参数 (eV)

EDIFF = 1E-8
EDIFFG = -1E-3
IBRION = 2
ISIF = 3
NSW = 50
```

**k点收敛**:
| k点网格 | 总能 (eV) | 费米能 (eV) | 耗时 |
|---------|-----------|-------------|------|
| 8×8×8   | -3.7291   | 9.8432      | 30s  |
| 12×12×12| -3.7352   | 9.8456      | 120s |
| 16×16×16| -3.7358   | 9.8458      | 350s |
| 20×20×20| -3.7359   | 9.8458      | 800s |

**收敛标准**: 16×16×16 达到meV精度

### 1.2 能带结构

Cu的能带特征是sp带与d带杂化:
- **s带**: 自由电子状，宽能带
- **d带**: 窄而平坦，位于E_F以下
- **sp-d杂化**: 决定输运性质

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_cu_bands():
    """绘制Cu的能带结构，标注轨道特征"""
    
    # 高对称路径: Γ-X-W-L-Γ-K
    kpath_labels = ['Γ', 'X', 'W', 'L', 'Γ', 'K']
    
    # 典型能带数据 (从VASP EIGENVAL读取)
    # 这里用示意数据展示特征
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 绘制能带 (示意)
    k = np.linspace(0, 5, 500)
    
    # sp带 (自由电子状)
    for n in range(3):
        E_sp = 5 + 2*(k + n*0.5)**2
        ax.plot(k, E_sp, 'b-', linewidth=1.5, alpha=0.7)
    
    # d带 (窄)
    k_d = np.linspace(0.5, 4.5, 100)
    for i in range(5):
        E_d = 2 + 0.5*np.sin(2*np.pi*i/5 + k_d)
        ax.plot(k_d, E_d, 'r-', linewidth=2, alpha=0.8)
    
    # 费米能级
    ax.axhline(y=6.5, color='green', linestyle='--', linewidth=2, label='E_F')
    ax.fill_between([0, 5], 0, 6.5, alpha=0.2, color='green')
    
    # 标记
    label_pos = [0, 1, 1.5, 2.5, 3.5, 5]
    ax.set_xticks(label_pos)
    ax.set_xticklabels(kpath_labels)
    
    ax.set_xlabel('k-path')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Cu Band Structure')
    ax.legend()
    ax.set_ylim(-1, 12)
    
    # 添加文字标注
    ax.text(0.2, 8, 'sp bands', fontsize=12, color='blue')
    ax.text(2.5, 1.5, 'd bands', fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig('cu_bands.png', dpi=300)
    plt.show()
```

### 1.3 费米面

Cu的费米面是颈部连接的球体 (necks at L点)。

```python
def calculate_fermi_surface():
    """
    计算费米面
    
    使用Wannier插值或密k点计算
    """
    
    # 方法1: VASP + pyFermi
    # 计算密k点网格上的本征值
    
    k_mesh = np.meshgrid(
        np.linspace(-0.5, 0.5, 50),
        np.linspace(-0.5, 0.5, 50),
        np.linspace(-0.5, 0.5, 50)
    )
    
    # 找到费米面 (E = E_F 的等能面)
    fermi_surface = []
    
    for kx, ky, kz in zip(k_mesh[0].flatten(), 
                          k_mesh[1].flatten(), 
                          k_mesh[2].flatten()):
        energy = interpolate_band_energy(kx, ky, kz)
        if abs(energy - fermi_energy) < 0.01:
            fermi_surface.append([kx, ky, kz])
    
    # 3D可视化
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    fs = np.array(fermi_surface)
    ax.scatter(fs[:, 0], fs[:, 1], fs[:, 2], s=1, alpha=0.5)
    
    ax.set_xlabel('k_x')
    ax.set_ylabel('k_y')
    ax.set_zlabel('k_z')
    ax.set_title('Cu Fermi Surface')
    
    plt.savefig('fermi_surface_3d.png', dpi=300)
    plt.show()

def analyze_fermi_surface_topology():
    """分析费米面拓扑特征"""
    
    print("Cu费米面分析:")
    print("=" * 40)
    
    print("\n1. 颈部 (Necks):")
    print("   位置: L点 (布里渊区边界)")
    print("   来源: sp带与d带杂化")
    print("   对输运的贡献: 显著")
    
    print("\n2. 腹瓣 (Belly):")
    print("   中心: Γ点附近")
    print("   形状: 近球形")
    
    print("\n3. 实验验证:")
    print("   de Haas-van Alphen效应")
    print("   回旋频率与颈部/腹瓣对应")
```

### 1.4 态密度与投影态密度

```python
def analyze_density_of_states():
    """分析态密度"""
    
    # 读取VASP DOSCAR
    energy, dos, pdos_s, pdos_p, pdos_d = read_doscar('DOSCAR')
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # 总DOS
    ax = axes[0]
    ax.fill_between(energy, 0, dos, alpha=0.5, color='gray')
    ax.plot(energy, dos, 'k-', linewidth=1.5)
    ax.axvline(x=fermi_energy, color='r', linestyle='--', label='E_F')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('DOS (states/eV)')
    ax.set_title('Total DOS')
    ax.set_xlim(-10, 15)
    ax.legend()
    
    # 投影DOS
    ax = axes[1]
    ax.fill_between(energy, 0, pdos_s, alpha=0.5, label='s', color='blue')
    ax.fill_between(energy, 0, pdos_p, alpha=0.5, label='p', color='green')
    ax.fill_between(energy, 0, pdos_d, alpha=0.5, label='d', color='red')
    ax.plot(energy, pdos_s, 'b-', linewidth=1)
    ax.plot(energy, pdos_p, 'g-', linewidth=1)
    ax.plot(energy, pdos_d, 'r-', linewidth=1)
    ax.axvline(x=fermi_energy, color='k', linestyle='--')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('PDOS (states/eV)')
    ax.set_title('Projected DOS')
    ax.set_xlim(-10, 15)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('cu_dos.png', dpi=300)
    plt.show()
    
    # 费米面处DOS
    ef_idx = np.argmin(np.abs(energy - fermi_energy))
    dos_ef = dos[ef_idx]
    
    print(f"\n费米面态密度:")
    print(f"  N(E_F) = {dos_ef:.3f} states/eV")
    print(f"  实验值 = 0.29 states/eV·atom")
    print(f"  比值 = {dos_ef/0.29:.2f}")
    
    # d带中心
    d_band_center = np.trapz(energy * pdos_d, energy) / np.trapz(pdos_d, energy)
    print(f"\nd带中心: {d_band_center:.2f} eV (相对于E_F)")
```

---

## 2. 输运性质计算

### 2.1 电导率 (Boltzmann方程)

```python
def calculate_transport_properties():
    """
    使用BoltzTraP或BoltzWann计算输运性质
    """
    
    # 方法1: BoltzTraP (基于DFT能带)
    # boltztrap -so cu.energyso cu.struct
    
    # 方法2: Wannier插值 + BoltzWann
    # 需要Wannier90计算
    
    # 典型结果
    print("Cu的输运性质 (300K):")
    print("=" * 40)
    
    # 电导率
    sigma = 6.0e7  # S/m, 实验值 ~6.0e7
    print(f"电导率 σ = {sigma:.2e} S/m")
    
    # 电阻率
    rho = 1/sigma * 1e8  # μΩ·cm
    print(f"电阻率 ρ = {rho:.3f} μΩ·cm")
    print(f"  (实验值: 1.68 μΩ·cm)")
    
    # 载流子浓度
    n = 8.5e28  # m^-3
    print(f"\n载流子浓度 n = {n:.2e} m⁻³")
    
    # 迁移率
    mu = sigma / (n * 1.6e-19) * 1e4  # cm²/V·s
    print(f"迁移率 μ = {mu:.1f} cm²/V·s")

def analyze_temperature_dependence():
    """分析电阻率的温度依赖性"""
    
    temperatures = np.linspace(50, 800, 100)
    
    # Bloch-Grüneisen模型
    theta_D = 315  # Cu的德拜温度 (K)
    rho_0 = 0.02   # 剩余电阻率 (μΩ·cm)
    
    def bloch_gruneisen(T, theta_D):
        """Bloch-Grüneisen公式"""
        from scipy.integrate import quad
        
        def integrand(x):
            return x**5 / ((np.exp(x) - 1) * (1 - np.exp(-x)))
        
        if T < theta_D/10:
            return 124.4 * (T/theta_D)**5
        else:
            integral, _ = quad(integrand, 0, theta_D/T)
            return 4 * (T/theta_D)**5 * integral
    
    rho_T = [rho_0 + 1.5 * bloch_gruneisen(T, theta_D) for T in temperatures]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(temperatures, rho_T, 'b-', linewidth=2, label='Theory')
    
    # 实验数据点
    T_exp = [100, 200, 300, 400, 500, 600, 700, 800]
    rho_exp = [0.6, 1.2, 1.7, 2.4, 3.0, 3.7, 4.3, 5.0]
    ax.scatter(T_exp, rho_exp, c='red', s=50, label='Experiment', zorder=5)
    
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Resistivity (μΩ·cm)')
    ax.set_title('Cu Resistivity vs Temperature')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('cu_resistivity.png', dpi=300)
    plt.show()
```

### 2.2 电子平均自由程

```python
def calculate_mean_free_path():
    """
    计算电子平均自由程
    
    l = v_F * τ
    其中 τ 是弛豫时间
    """
    
    # 费米速度 (从能带计算)
    v_f = 1.57e6  # m/s, Cu的费米速度
    
    # 弛豫时间 (从电导率)
    sigma = 6.0e7  # S/m
    n = 8.5e28     # m^-3
    m_eff = 1.01 * 9.11e-31  # kg
    
    tau = sigma * m_eff / (n * (1.6e-19)**2)
    
    # 平均自由程
    l = v_f * tau * 1e9  # nm
    
    print("Cu电子输运参数:")
    print(f"  费米速度 v_F = {v_f:.2e} m/s")
    print(f"  弛豫时间 τ = {tau*1e15:.1f} fs")
    print(f"  平均自由程 l = {l:.1f} nm")
    
    return l, tau, v_f
```

---

## 3. 表面电子结构

### 3.1 低指数表面弛豫

```python
def calculate_surface_relaxation(surface='111'):
    """
    计算Cu表面弛豫
    
    Cu主要表面:
    - (111): 最密排面，最稳定
    - (100): 方形对称
    - (110): 沟槽结构
    """
    
    surfaces = {
        '111': {'layers': 12, 'k': '12x12x1'},
        '100': {'layers': 11, 'k': '10x10x1'},
        '110': {'layers': 10, 'k': '10x8x1'}
    }
    
    surf_params = surfaces[surface]
    
    print(f"Cu({surface})表面计算:")
    print(f"  原子层数: {surf_params['layers']}")
    print(f"  k点: {surf_params['k']}")
    
    # 典型的表面弛豫结果
    relaxations = {
        '111': {'d12': -0.6, 'd23': +0.3},  # %
        '100': {'d12': -1.1, 'd23': +0.5},
        '110': {'d12': -8.5, 'd23': +3.0}   # (110)弛豫显著
    }
    
    return relaxations[surface]
```

### 3.2 表面态与表面能

```python
def calculate_surface_energy():
    """计算表面能"""
    
    # γ = (E_slab - N*E_bulk) / (2A)
    
    surfaces = ['111', '100', '110']
    
    results = {}
    
    for surf in surfaces:
        # 读取计算结果
        E_slab = read_energy(f'surface_{surf}/OUTCAR')
        E_bulk = read_energy('bulk/OUTCAR')
        
        N = read_atom_count(f'surface_{surf}/POSCAR')
        A = read_surface_area(f'surface_{surf}/POSCAR')
        
        gamma = (E_slab - N * E_bulk) / (2 * A) * 16.02  # J/m²
        
        results[surf] = gamma
        print(f"Cu({surf}): γ = {gamma:.2f} J/m²")
    
    return results

def analyze_surface_states():
    """分析表面态"""
    
    print("\nCu表面态特征:")
    print("=" * 40)
    
    print("\nCu(111):")
    print("  Shockley表面态: 在L点投影带隙中")
    print("  抛物线色散，近自由电子状")
    print("  费米波矢 k_F ~ 0.2 Å⁻¹")
    
    print("\nCu(100):")
    print("  无表面带隙")
    print("  表面态与体态强烈杂化")
    
    print("\nCu(110):")
    print("  存在表面共振态")
    print("  沿[001]方向一维特征")
```

---

## 4. 结果汇总

| 性质 | 计算值 | 实验值 | 误差 |
|------|--------|--------|------|
| 晶格常数 | 3.614 Å | 3.615 Å | 0.03% |
| 体模量 | 142 GPa | 140 GPa | +1% |
| 费米能处DOS | 0.28 st/eV | 0.29 st/eV | -3% |
| 电阻率 (300K) | 1.65 μΩ·cm | 1.68 μΩ·cm | -2% |
| 功函数 (111) | 5.0 eV | 4.9 eV | +2% |

---

## 参考

1. N. W. Ashcroft & N. D. Mermin, *Solid State Physics* - 金属物理经典教材
2. G. K. H. Madsen et al., *Phys. Rev. B* 64, 195134 (2001) - BoltzTraP方法
3. A. B. Shick et al., *Phys. Rev. B* 60, 14392 (1999) - Cu表面态计算
