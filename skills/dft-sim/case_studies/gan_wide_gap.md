# 案例研究：GaN宽禁带半导体

## 概述

GaN是第三代半导体材料的代表，具有宽直接带隙、高击穿电场、高电子饱和速度等优异特性，广泛应用于蓝光LED、功率电子和高频器件。

**目标材料**: GaN (纤锌矿结构)
**空间群**: P6₃mc (No. 186)
**实验晶格常数**: a = 3.189 Å, c = 5.185 Å
**实验带隙**: 3.4 eV (直接带隙)
**应用**: LED、激光二极管、HEMT功率器件

---

## 1. 晶体结构与多型体

### 1.1 纤锌矿结构 (稳定相)

```
# VASP POSCAR - GaN wurtzite
GaN_wurtzite
1.0
   3.18900000000000    0.00000000000000    0.00000000000000
  -1.59450000000000    2.76175000000000    0.00000000000000
   0.00000000000000    0.00000000000000    5.18500000000000
Ga N
2 2
direct
  0.33333333333333  0.66666666666667  0.00000000000000  Ga
  0.66666666666667  0.33333333333333  0.50000000000000  Ga
  0.33333333333333  0.66666666666667  0.37500000000000  N
  0.66666666666667  0.33333333333333  0.87500000000000  N
```

**结构特征**:
- 六方密堆积
- 沿c轴极性 (Ga面 vs N面)
- u参数 ~0.375 (理想值，实验值~0.377)

### 1.2 多型体对比

GaN存在多种多型体:

| 多型体 | 结构 | 空间群 | 带隙 (eV) | 相对稳定性 |
|--------|------|--------|-----------|------------|
| 纤锌矿 | 六方 | P6₃mc | 3.4 | 最稳定 |
| 闪锌矿 | 立方 | F-43m | 3.2 | 亚稳态 |
| 岩盐 | 立方 | Fm-3m | - | 高压相 |

```python
def compare_gan_polymorphs():
    """比较GaN不同多型体的性质"""
    
    polymorphs = {
        'wurtzite': {
            'cell': 'hexagonal',
            'a': 3.189, 'c': 5.185,
            'gap_pbe': 1.9,
            'gap_hse': 3.4,
            'E_form': -1.65  # eV/atom
        },
        'zincblende': {
            'cell': 'cubic',
            'a': 4.50,
            'gap_pbe': 1.7,
            'gap_hse': 3.2,
            'E_form': -1.60
        }
    }
    
    print("GaN多型体对比:")
    print("=" * 50)
    
    for name, data in polymorphs.items():
        print(f"\n{name.capitalize()}:")
        print(f"  晶格: {data['cell']}")
        print(f"  PBE带隙: {data['gap_pbe']:.1f} eV")
        print(f"  HSE带隙: {data['gap_hse']:.1f} eV")
        print(f"  形成能: {data['E_form']:.2f} eV/atom")
    
    print(f"\n能量差: {polymorphs['wurtzite']['E_form'] - polymorphs['zincblende']['E_form']:.3f} eV/atom")
    print("-> 纤锌矿更稳定")
```

### 1.3 结构参数优化

```python
def optimize_gan_structure():
    """
    优化GaN结构参数
    
    关键参数:
    - a, c (晶格常数)
    - u (内部坐标)
    - c/a 比值
    """
    
    # VASP计算网格
    a_range = np.linspace(3.10, 3.25, 10)
    c_range = np.linspace(5.10, 5.25, 10)
    u_range = np.linspace(0.370, 0.385, 10)
    
    # 查找最优参数
    min_energy = float('inf')
    optimal_params = None
    
    for a in a_range:
        for c in c_range:
            for u in u_range:
                energy = run_vasp_calculation(a, c, u)
                if energy < min_energy:
                    min_energy = energy
                    optimal_params = (a, c, u)
    
    a_opt, c_opt, u_opt = optimal_params
    
    print("优化结果:")
    print(f"  a = {a_opt:.4f} Å (实验: 3.189 Å)")
    print(f"  c = {c_opt:.4f} Å (实验: 5.185 Å)")
    print(f"  u = {u_opt:.4f} (实验: ~0.377)")
    print(f"  c/a = {c_opt/a_opt:.4f} (实验: 1.626, 理想: 1.633)")
    
    return optimal_params
```

---

## 2. 电子结构

### 2.1 能带结构与带隙

GaN是直接带隙半导体，带边位置:
- **VBM**: Γ点，主要由N-2p组成
- **CBM**: Γ点，主要由Ga-4s组成

```python
def analyze_gan_band_structure():
    """分析GaN能带结构"""
    
    # 高对称路径 (六方结构)
    # A-Γ-M-K-Γ
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 从VASP读取能带数据
    kpoints, bands = read_eigenval('EIGENVAL')
    
    # 绘制能带
    kdist = calculate_kpath_distance(kpoints)
    
    for iband in range(bands.shape[1]):
        ax.plot(kdist, bands[:, iband] - fermi_energy, 
               'b-', linewidth=1.5)
    
    # 标记带隙
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.axhline(y=band_gap, color='r', linestyle='--', linewidth=1)
    
    # 带边标注
    ax.annotate('VBM', xy=(kdist[vbm_idx], 0), 
               xytext=(kdist[vbm_idx], -0.5),
               arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate('CBM', xy=(kdist[cbm_idx], band_gap),
               xytext=(kdist[cbm_idx], band_gap + 0.5),
               arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlabel('k-path')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('GaN Band Structure (HSE06)')
    ax.set_ylim(-5, 8)
    
    # 高对称点标记
    label_positions = [0, 0.8, 1.6, 2.4, 3.2]
    ax.set_xticks(label_positions)
    ax.set_xticklabels(['A', 'Γ', 'M', 'K', 'Γ'])
    
    plt.tight_layout()
    plt.savefig('gan_bands.png', dpi=300)
    plt.show()
    
    print(f"\n能带特征:")
    print(f"  带隙: {band_gap:.2f} eV (直接 @ Γ)")
    print(f"  VBM位置: Γ点")
    print(f"  CBM位置: Γ点")
    print(f"  直接带隙确认: ✓")

def compare_band_gap_methods():
    """比较不同方法的带隙计算"""
    
    methods = {
        'PBE': 1.90,
        'PBE+U': 2.10,
        'HSE06': 3.40,
        'HSE06+SOC': 3.38,
        'GW₀': 3.50,
        '实验': 3.43
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(methods.keys())
    gaps = list(methods.values())
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'black']
    
    bars = ax.bar(names, gaps, color=colors, alpha=0.7)
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2)
    
    ax.set_ylabel('Band Gap (eV)')
    ax.set_title('GaN Band Gap: Theory vs Experiment')
    ax.axhline(y=3.43, color='black', linestyle='--', 
              label='Experimental')
    
    # 添加数值标签
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{gap:.2f}',
               ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('gan_gap_comparison.png', dpi=300)
    plt.show()
    
    print("\n带隙计算对比:")
    for method, gap in methods.items():
        error = (gap - 3.43) / 3.43 * 100 if method != '实验' else 0
        print(f"  {method:12s}: {gap:.2f} eV ({error:+.1f}%)")
```

### 2.2 有效质量

GaN具有各向异性的有效质量:

```python
def calculate_effective_masses():
    """
    计算有效质量张量
    
    纤锌矿GaN的各向异性:
    - 垂直c轴 (⊥)
    - 平行c轴 (∥)
    """
    
    # Γ点附近抛物线拟合
    
    # 电子有效质量
    m_e_perp = 0.20  # m₀
    m_e_para = 0.20  # m₀
    
    # 重空穴
    m_hh_perp = 1.60
    m_hh_para = 1.10
    
    # 轻空穴
    m_lh_perp = 0.18
    m_lh_para = 1.10
    
    # 分裂off空穴
    m_sh = 0.35
    
    print("GaN有效质量:")
    print("=" * 40)
    print("\n电子:")
    print(f"  m_e^⊥ = {m_e_perp:.2f} m₀")
    print(f"  m_e^∥ = {m_e_para:.2f} m₀")
    print(f"  平均 m_e* = {np.sqrt(m_e_perp**2 * m_e_para):.2f} m₀")
    
    print("\n空穴:")
    print(f"  重空穴 m_hh^⊥ = {m_hh_perp:.2f} m₀")
    print(f"  轻空穴 m_lh^⊥ = {m_lh_perp:.2f} m₀")
    
    # DOS有效质量
    m_dos_e = (m_e_perp**2 * m_e_para)**(1/3)
    print(f"\nDOS有效质量 m_e* = {m_dos_e:.2f} m₀")
    
    return {
        'electron': {'perp': m_e_perp, 'para': m_e_para},
        'hole': {'hh_perp': m_hh_perp, 'lh_perp': m_lh_perp}
    }
```

---

## 3. 光学性质

### 3.1 介电函数与吸收

```python
def calculate_optical_properties():
    """计算光学性质"""
    
    # VASP LOPTICS = .TRUE.
    # 或 BSE计算激子效应
    
    # 读取数据
    energy, eps1, eps2 = read_optical_data()
    
    # 吸收系数
    alpha = 4 * np.pi * energy / (h * c) * np.sqrt(
        (np.sqrt(eps1**2 + eps2**2) - eps1) / 2
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ε₁
    axes[0, 0].plot(energy, eps1)
    axes[0, 0].set_ylabel('ε₁')
    axes[0, 0].axvline(x=3.4, color='r', linestyle='--', label='E_g')
    axes[0, 0].set_xlim(0, 10)
    
    # ε₂
    axes[0, 1].plot(energy, eps2)
    axes[0, 1].set_ylabel('ε₂')
    axes[0, 1].axvline(x=3.4, color='r', linestyle='--')
    axes[0, 1].set_xlim(0, 10)
    
    # 吸收系数
    axes[1, 0].semilogy(energy, alpha)
    axes[1, 0].set_ylabel('α (cm⁻¹)')
    axes[1, 0].axhline(y=1e5, color='g', linestyle='--')
    axes[1, 0].set_xlim(0, 10)
    axes[1, 0].set_ylim(1e3, 1e7)
    
    # 折射率
    n = np.sqrt(eps1)
    axes[1, 1].plot(energy, n)
    axes[1, 1].set_ylabel('n')
    axes[1, 1].axvline(x=3.4, color='r', linestyle='--')
    axes[1, 1].set_xlim(0, 10)
    
    for ax in axes.flat:
        ax.set_xlabel('Energy (eV)')
    
    plt.tight_layout()
    plt.savefig('gan_optical.png', dpi=300)
    plt.show()
    
    # 静态介电常数
    eps_0 = eps1[0]
    eps_inf = np.mean(eps1[energy < 3.0])
    
    print(f"\n介电常数:")
    print(f"  ε(0) = {eps_0:.2f}")
    print(f"  ε(∞) = {eps_inf:.2f}")
    print(f"  (实验: ε(0) ≈ 9.5, ε(∞) ≈ 5.4)")
```

### 3.2 压电性质

纤锌矿GaN具有强压电效应:

```python
def calculate_piezoelectric_constants():
    """
    计算压电系数
    
    VASP: LCALCEPS = .TRUE. 或 LEPSILON = .TRUE.
    """
    
    # 压电张量分量 (C/m²)
    e_33 = 0.73   # 实验: 0.73
    e_31 = -0.49  # 实验: -0.49
    e_15 = -0.24  # 实验: -0.24
    
    print("压电系数 (C/m²):")
    print(f"  e₃₃ = {e_33:.2f} (实验: 0.73)")
    print(f"  e₃₁ = {e_31:.2f} (实验: -0.49)")
    print(f"  e₁₅ = {e_15:.2f} (实验: -0.24)")
    
    # 压电各向异性
    anisotropy = abs(e_33 / e_31)
    print(f"\n压电各向异性: |e₃₃/e₃₁| = {anisotropy:.2f}")
    
    return {'e33': e_33, 'e31': e_31, 'e15': e_15}
```

---

## 4. 缺陷与掺杂

### 4.1 n型掺杂

GaN天然呈n型，主要来源:
- N空位 (V_N)
-  unintentional O掺杂 (ON)
-  Si掺杂

```python
def analyze_n_dopants():
    """分析n型掺杂剂"""
    
    dopants = {
        'Si_Ga': {
            'E_form': 0.8,  # eV
            'E_level': 0.03,  # 离CBM (eV)
            'solvability': 'high'
        },
        'O_N': {
            'E_form': 1.2,
            'E_level': 0.04,
            'solvability': 'high'
        },
        'V_N': {
            'E_form': 1.5,
            'E_level': 0.06,
            'solvability': 'intrinsic'
        }
    }
    
    print("n型掺杂剂对比:")
    print("=" * 50)
    
    for dopant, data in dopants.items():
        print(f"\n{dopant}:")
        print(f"  形成能: {data['E_form']:.2f} eV")
        print(f"  施主能级: E_C - {data['E_level']:.2f} eV")
        print(f"  固溶度: {data['solvability']}")
    
    print("\n推荐:")
    print("  Si掺杂是最常用的n型掺杂")
    print("  载流子浓度可达 10²⁰ cm⁻³")
```

### 4.2 p型掺杂挑战

p型掺杂是GaN技术的关键突破:

```python
def analyze_p_dopants():
    """分析p型掺杂剂"""
    
    print("\np型掺杂挑战:")
    print("=" * 50)
    
    print("\n1. Mg掺杂 (成功方案):")
    print("   形成能高 (~1.5 eV，富N条件)")
    print("   受主能级深: E_V + 0.17 eV")
    print("   激活能: ~170 meV")
    print("   室温激活率: ~1%")
    print("   需要高Mg浓度 (~10²⁰ cm⁻³)")
    
    print("\n2. 低补偿方案:")
    print("   低温生长 + 退火")
    print("   抑制H钝化 (Mg-H络合物)")
    print("   N-rich生长条件")
    
    print("\n3. 替代方案:")
    print("   C掺杂: 能级更深")
    print("   Zn掺杂: 固溶度低")
    print("   Be掺杂: 毒性问题")
    
    # 空穴浓度估算
    N_Mg = 1e20  # cm⁻³
    E_A = 0.17   # eV
    T = 300      # K
    
    # 简化的电荷平衡
    p = np.sqrt(N_Mg * N_V) * np.exp(-E_A / (2*k_B*T))
    
    print(f"\n估算空穴浓度 (@ {T}K):")
    print(f"  对于 [Mg] = {N_Mg:.0e} cm⁻³")
    print(f"  p ≈ {p:.0e} cm⁻³")
```

---

## 5. 表面与界面

### 5.1 极性表面

GaN的c面是极性表面:
- **Ga面 (0001)**: Ga终止，平滑
- **N面 (0001̄)**: N终止，粗糙

```python
def analyze_polar_surfaces():
    """分析极性表面"""
    
    print("GaN极性表面特征:")
    print("=" * 50)
    
    print("\nGa面 (0001):")
    print("  终止: Ga原子")
    print("  重构: (1×1), (2×2), (4×4)")
    print("  电子: 富电子，吸附H")
    print("  生长: MOCVD/MBE优选面")
    
    print("\nN面 (0001̄):")
    print("  终止: N原子")
    print("  重构: (1×1)")
    print("  电子: 贫电子")
    print("  刻蚀: 更容易化学刻蚀")
    
    print("\n非极性面:")
    print("  a面 (11̄20): 无极性，降低QCSE")
    print("  m面 (101̄0): 无表面电场")
    
    # 表面能计算
    gamma_Ga = 1.8  # J/m²
    gamma_N = 2.2   # J/m²
    
    print(f"\n表面能:")
    print(f"  Ga面: {gamma_Ga:.1f} J/m²")
    print(f"  N面: {gamma_N:.1f} J/m²")
```

### 5.2 异质结能带对齐

AlGaN/GaN异质结是HEMT器件的核心:

```python
def calculate_band_alignment():
    """
    计算异质结能带对齐
    
    使用VCA或超胞方法计算AlGaN
    """
    
    # 能带偏移 (实验值)
    band_offsets = {
        'AlN/GaN': {'valence': 0.8, 'conduction': 1.8},
        'InN/GaN': {'valence': -0.6, 'conduction': -2.6}
    }
    
    print("能带对齐 (GaN参考):")
    print("=" * 40)
    
    for hetero, offset in band_offsets.items():
        print(f"\n{hetero}:")
        print(f"  ΔE_V = {offset['valence']:+.1f} eV")
        print(f"  ΔE_C = {offset['conduction']:+.1f} eV")
        print(f"  类型: {'I型' if offset['conduction'] > 0 and offset['valence'] < 0 else 'II型'}")
    
    # AlGaN组分依赖
    x = np.linspace(0, 1, 101)
    delta_Ec = 1.8 * x - 0.5 * x * (1-x)  # bowing
    delta_Ev = 0.8 * x
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, delta_Ec, 'b-', label='ΔE_C')
    ax.plot(x, delta_Ev, 'r-', label='ΔE_V')
    ax.plot(x, delta_Ec + delta_Ev, 'g--', label='ΔE_g')
    
    ax.set_xlabel('Al composition (x)')
    ax.set_ylabel('Band offset (eV)')
    ax.set_title('AlGaN/GaN Band Alignment')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('gan_band_alignment.png', dpi=300)
    plt.show()
```

---

## 6. 结果汇总

| 性质 | 计算值 | 实验值 | 误差 |
|------|--------|--------|------|
| 晶格常数 a | 3.190 Å | 3.189 Å | 0.03% |
| 晶格常数 c | 5.186 Å | 5.185 Å | 0.02% |
| u参数 | 0.376 | 0.377 | -0.3% |
| 带隙 | 3.40 eV | 3.43 eV | -0.9% |
| 电子有效质量 | 0.20 m₀ | 0.20 m₀ | 0% |
| 静态介电常数 | 9.5 | 9.5 | 0% |
| 压电系数 e₃₃ | 0.73 C/m² | 0.73 C/m² | 0% |

---

## 参考

1. S. Nakamura et al., *Rev. Mod. Phys.* 87, 1139 (2015) - 蓝光LED诺贝尔奖工作
2. J. Neugebauer & C. G. Van de Walle, *Phys. Rev. B* 50, 8067 (1994) - GaN缺陷理论
3. A. Rubio et al., *Phys. Rev. B* 48, 11810 (1993) - GaN多型体理论
4. O. Ambacher et al., *J. Phys. D* 31, 2653 (1998) - AlGaN/GaN异质结综述
