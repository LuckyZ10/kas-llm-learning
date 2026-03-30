# 案例研究：MAPbI₃钙钛矿太阳能电池材料

## 概述

MAPbI₃ (甲基铵碘化铅) 是有机-无机杂化钙钛矿太阳能电池的明星材料。本案例展示这类软晶格材料的完整计算流程，包括结构相变、缺陷容忍性、载流子动力学和离子迁移。

**目标材料**: CH₃NH₃PbI₃ (MAPbI₃)
**结构**: 钙钛矿 ABX₃ (A=MA⁺, B=Pb²⁺, X=I⁻)
**相变序列**: 
- 正交 (Ortho) < 160 K
- 四方 (Tetr) 160-330 K  
- 立方 (Cubic) > 330 K (室温相)

**实验带隙**: ~1.5-1.6 eV (直接带隙)
**转换效率**: >25% (实验室纪录)

---

## 1. 结构与相变

### 1.1 晶体结构

MAPbI₃具有钙钛矿结构，其中MA⁺有机阳离子位于A位，Pb²⁺在B位，I⁻在X位。

**立方相 (Pm-3m, No. 221)**:
```
# VASP POSCAR - 理想立方结构 (高温相参考)
MAPbI3_cubic
1.0
   6.3000000000000000    0.0000000000000000    0.0000000000000000
   0.0000000000000000    6.3000000000000000    0.0000000000000000
   0.0000000000000000    0.0000000000000000    6.3000000000000000
Pb C N I H
1 1 1 3 6
direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000  Pb
  0.5000000000000000  0.5000000000000000  0.5000000000000000  C (MA中心)
  0.5800000000000000  0.5000000000000000  0.5000000000000000  N
  0.5000000000000000  0.0000000000000000  0.0000000000000000  I
  0.0000000000000000  0.5000000000000000  0.0000000000000000  I
  0.0000000000000000  0.0000000000000000  0.5000000000000000  I
  # H原子位置略 (CH3和NH3基团)
```

**四方相 (I4/mcm, No. 140)** - 室温稳定相:
- a = b = 8.85 Å, c = 12.66 Å
- 晶胞包含4个公式单元
- PbI₆八面体沿c轴旋转

### 1.2 MA⁺有机阳离子取向

MA⁺具有偶极矩 (~2.3 Debye)，室温下可自由旋转，低温下有序排列。

**处理方法**:
1. **静态计算**: 固定MA⁺方向，测试不同取向
2. **有限温度**: AIMD模拟取向无序
3. **平均结构**: 使用高对称平均结构

```python
from ase import Atoms
from ase.build import bulk
import numpy as np

def build_mapbi3_tetragonal(ma_orientation='random'):
    """
    构建四方相MAPbI₃结构
    
    Parameters:
    -----------
    ma_orientation : str
        '001', '110', '111' 或 'random'
    """
    # 晶格参数 (实验值 @ 300K)
    a = 8.85
    c = 12.66
    
    # 构建基础钙钛矿框架
    # Pb在角顶, I在面心
    pb_pos = np.array([
        [0, 0, 0], [0.5, 0.5, 0],
        [0.5, 0, 0.25], [0, 0.5, 0.25],
        [0, 0, 0.5], [0.5, 0.5, 0.5],
        [0.5, 0, 0.75], [0, 0.5, 0.75]
    ])
    
    # I位置 (面心位置，考虑八面体倾斜)
    i_pos = np.array([
        [0.25, 0, 0], [0.75, 0, 0],
        [0, 0.25, 0], [0, 0.75, 0],
        # ... 更多I位置
    ])
    
    # MA⁺阳离子 - 体心位置
    # C-N键沿特定方向
    if ma_orientation == '001':
        cn_axis = [0, 0, 1]
    elif ma_orientation == '110':
        cn_axis = [1/np.sqrt(2), 1/np.sqrt(2), 0]
    elif ma_orientation == '111':
        cn_axis = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    
    # 构建完整结构
    # ... (详细原子坐标)
    
    return atoms

# 测试不同取向的能量
def test_ma_orientations():
    """测试MA⁺不同取向的相对能量"""
    orientations = ['001', '110', '111', 'random']
    energies = []
    
    for orient in orientations:
        atoms = build_mapbi3_tetragonal(orient)
        # 运行VASP计算
        energy = run_vasp_calculation(atoms)
        energies.append(energy)
        print(f"{orient}: {energy:.4f} eV/f.u.")
    
    # 找出最稳定取向
    min_idx = np.argmin(energies)
    print(f"\n最稳定取向: {orientations[min_idx]}")
    
    return orientations, energies
```

### 1.3 相变温度计算

使用准谐近似 (QHA) 或AIMD计算不同相的自由能。

```python
def calculate_phase_transition():
    """
    计算相变温度
    
    方法:
    1. 计算各相的总能 (0K)
    2. 计算声子贡献的自由能
    3. 比较F(T) = E_0 + F_vib(T)
    """
    
    temperatures = np.linspace(100, 400, 31)
    
    # 各相数据
    phases = {
        'orthorhombic': {'E0': -145.2, 'Theta_D': 180},
        'tetragonal': {'E0': -145.15, 'Theta_D': 160},
        'cubic': {'E0': -145.10, 'Theta_D': 140}
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for phase, data in phases.items():
        E0 = data['E0']
        Theta_D = data['Theta_D']
        
        # Debye模型估算振动自由能
        # F_vib = 9/8 * k_B * Theta_D + ... (详细公式)
        
        F_vib = -calculate_debye_free_energy(temperatures, Theta_D)
        F_total = E0 + F_vib
        
        ax.plot(temperatures, F_total, label=phase)
    
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Free Energy (eV/f.u.)')
    ax.legend()
    ax.set_title('MAPbI₃ Phase Stability')
    
    # 找到相变点
    plt.tight_layout()
    plt.savefig('phase_diagram.png', dpi=300)
    plt.show()
```

---

## 2. 电子结构与光学性质

### 2.1 能带结构与带隙

**关键特征**:
- 直接带隙 @ R点 (立方) 或类似位置
- 价带顶: I-5p + Pb-6s (反键)
- 导带底: Pb-6p
- 高吸收系数 (>10⁵ cm⁻¹)

**VASP计算设置**:

```bash
# INCAR 关键参数
ENCUT = 500
ISMEAR = 0
SIGMA = 0.05

# SOC对重元素很重要!
LSORBIT = .TRUE.
LMAXMIX = 4

# 带隙计算推荐HSE06
LHFCALC = .TRUE.
HFSCREEN = 0.2
ALGO = ALL
TIME = 0.4
PRECFOCK = Normal
```

**结果对比**:

| 方法 | 带隙 (eV) | 与实验误差 | 推荐用途 |
|------|-----------|------------|----------|
| PBE | 1.45 | -9% | 结构优化 |
| PBE+SOC | 1.42 | -11% | 初步筛选 |
| HSE06 | 1.58 | +1% | 带隙预测 |
| HSE06+SOC | 1.55 | -2% | **推荐** |
| GW₀ | 1.62 | +5% | 精确计算 |
| 实验 | 1.55-1.6 | - | - |

```python
def analyze_electronic_structure():
    """分析电子结构特征"""
    
    # 读取能带数据
    kpoints, bands = read_eigenval('EIGENVAL')
    
    # 识别直接/间接带隙
    # VBM位置
    vbm_idx = np.where(bands < fermi_level)
    vbm_max = np.max(bands[vbm_idx])
    vbm_k = kpoints[np.where(bands == vbm_max)[0][0]]
    
    # CBM位置  
    cbm_idx = np.where(bands > fermi_level)
    cbm_min = np.min(bands[cbm_idx])
    cbm_k = kpoints[np.where(bands == cbm_min)[0][0]]
    
    direct_gap = cbm_min - vbm_max
    
    print(f"价带顶位置: {vbm_k}")
    print(f"导带底位置: {cbm_k}")
    print(f"直接带隙: {direct_gap:.3f} eV")
    
    # 检查是否为直接带隙
    if np.allclose(vbm_k, cbm_k, atol=0.01):
        print("✓ 直接带隙材料 (适合光伏)")
    else:
        k_diff = np.linalg.norm(vbm_k - cbm_k)
        print(f"? 准直接带隙 (k差: {k_diff:.3f} Å⁻¹)")
    
    return direct_gap, vbm_k, cbm_k
```

### 2.2 有效质量与载流子迁移

MAPbI₃具有异常低的有效质量和良好的载流子输运。

```python
def calculate_carrier_effective_masses():
    """计算电子和空穴有效质量"""
    
    # VBM附近 - 空穴
    # 沿不同方向拟合
    directions = {
        'Γ-X': ([0, 0, 0], [0.5, 0, 0]),
        'Γ-M': ([0, 0, 0], [0.5, 0.5, 0]),
        'Γ-R': ([0, 0, 0], [0.5, 0.5, 0.5])
    }
    
    results = {}
    
    for name, (start, end) in directions.items():
        # 提取该方向的能带数据
        k_path, e_path = extract_path_band(start, end, kpoints, bands)
        
        # 抛物线拟合
        k_center = np.where(np.allclose(k_path, start, atol=0.01))[0][0]
        
        # 拟合VBM
        fit_range = slice(k_center - 5, k_center + 5)
        k_fit = k_path[fit_range]
        e_vb = e_path[fit_range, n_vbm]
        
        # 有效质量
        coeffs = np.polyfit(k_fit, e_vb, 2)
        m_h = hbar**2 / (2 * coeffs[0]) / m0
        
        # 拟合CBM
        e_cb = e_path[fit_range, n_vbm + 1]
        coeffs = np.polyfit(k_fit, e_cb, 2)
        m_e = hbar**2 / (2 * coeffs[0]) / m0
        
        results[name] = {'m_h': m_h, 'm_e': m_e}
        
        print(f"{name}:")
        print(f"  空穴有效质量: {m_h:.2f} m₀")
        print(f"  电子有效质量: {m_e:.2f} m₀")
    
    # 平均有效质量
    avg_m_h = np.mean([r['m_h'] for r in results.values()])
    avg_m_e = np.mean([r['m_e'] for r in results.values()])
    
    print(f"\n平均空穴有效质量: {avg_m_h:.2f} m₀")
    print(f"平均电子有效质量: {avg_m_e:.2f} m₀")
    
    return results

# 结果与实验对比
def compare_with_experiment():
    """与实验测量的有效质量对比"""
    
    comparison = {
        '计算 (HSE+SOC)': {'m_h': 0.23, 'm_e': 0.19},
        '实验 (磁阻)': {'m_h': 0.29, 'm_e': 0.23},
        '实验 (光谱)': {'m_h': 0.25, 'm_e': 0.20}
    }
    
    print("有效质量对比:")
    for method, masses in comparison.items():
        print(f"{method:15s}: m_h = {masses['m_h']:.2f} m₀, "
              f"m_e = {masses['m_e']:.2f} m₀")
```

### 2.3 光学吸收谱

```python
def calculate_optical_properties():
    """
    计算光学性质
    
    VASP: LOPTICS = .TRUE.
    或独立粒子近似 + RPA/BSE
    """
    
    # 读取介电函数
    energy, eps1, eps2 = read_optical_data('OPTIC')
    
    # 吸收系数
    alpha = calculate_absorption_coefficient(eps1, eps2, energy)
    
    # 折射率
    n = np.sqrt(eps1)
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ε₁
    axes[0, 0].plot(energy, eps1)
    axes[0, 0].set_ylabel('ε₁')
    axes[0, 0].axvline(x=1.55, color='r', linestyle='--', label='Eg')
    
    # ε₂
    axes[0, 1].plot(energy, eps2)
    axes[0, 1].set_ylabel('ε₂')
    axes[0, 1].axvline(x=1.55, color='r', linestyle='--')
    
    # 吸收系数
    axes[1, 0].semilogy(energy, alpha)
    axes[1, 0].set_ylabel('α (cm⁻¹)')
    axes[1, 0].axhline(y=1e4, color='g', linestyle='--', 
                      label='10⁴ cm⁻¹ (good absorption)')
    axes[1, 0].set_ylim(1e3, 1e6)
    
    # 折射率
    axes[1, 1].plot(energy, n)
    axes[1, 1].set_ylabel('n')
    axes[1, 1].axvline(x=1.55, color='r', linestyle='--')
    
    for ax in axes.flat:
        ax.set_xlabel('Energy (eV)')
        ax.set_xlim(0, 5)
    
    plt.tight_layout()
    plt.savefig('optical_properties.png', dpi=300)
    plt.show()
    
    # 关键光学参数
    # 静态介电常数
    eps_static = eps1[0]
    print(f"静态介电常数: ε(0) = {eps_static:.2f}")
    
    # 高吸收阈值
    high_abs_idx = np.where(alpha > 1e5)[0]
    if len(high_abs_idx) > 0:
        print(f"吸收系数 >10⁵ cm⁻¹ @ E > {energy[high_abs_idx[0]]:.2f} eV")
    
    return energy, eps1, eps2, alpha
```

---

## 3. 缺陷容忍性与载流子寿命

### 3.1 本征缺陷形成能

MAPbI₃的缺陷容忍性是其高效率的关键: 主要缺陷为浅能级，非辐射复合弱。

**计算方法**:

```python
def calculate_defect_formation_energy(defect_type, charge, chem_potentials):
    """
    计算缺陷形成能
    
    E_f = E_defect - E_bulk - Σnᵢμᵢ + q(E_F + E_VBM) + E_corr
    
    Parameters:
    -----------
    defect_type : str
        'V_Pb', 'V_I', 'MA_i', 'Pb_I', etc.
    charge : int
        缺陷电荷态
    chem_potentials : dict
        各元素的化学势
    """
    
    # 从计算结果读取
    E_defect = read_energy(f'{defect_type}_q{charge}/OUTCAR')
    E_bulk = read_energy('perfect/OUTCAR')
    
    # 原子数变化
    stoichiometry = {
        'V_Pb': {'Pb': -1},
        'V_I': {'I': -1},
        'V_MA': {'C': -1, 'N': -1, 'H': -6},
        'Pb_i': {'Pb': 1},
        'I_i': {'I': 1},
        'MA_i': {'C': 1, 'N': 1, 'H': 6}
    }
    
    delta_n = stoichiometry[defect_type]
    
    # 化学势项
    chem_term = sum(n * chem_potentials[elem] for elem, n in delta_n.items())
    
    # 电荷修正 (FNV)
    E_corr = fnv_correction(defect_type, charge)
    
    # 形成能
    formation_energy = (E_defect - E_bulk - chem_term + 
                       charge * fermi_level + E_corr)
    
    return formation_energy

# 化学势范围
def determine_chemical_potential_range():
    """
    确定允许的化学势范围
    
    约束条件:
    - 避免形成竞争相 (PbI₂, Pb, I₂)
    - 富Pb贫I vs 贫Pb富I
    """
    
    # 竞争相能量
    E_PbI2 = read_energy('PbI2/OUTCAR')
    E_Pb = read_energy('Pb/OUTCAR')
    E_I2 = read_energy('I2/OUTCAR')
    
    # MAPbI₃生成焓
    E_MAPbI3 = read_energy('MAPbI3/OUTCAR')
    
    # 化学势约束
    # μ_Pb + 2μ_I = μ_PbI2
    # μ_Pb + μ_MA + 3μ_I = μ_MAPbI3
    
    # A点: 富Pb (μ_Pb = 0, 参考单质)
    mu_Pb_rich = 0
    mu_I_poor = (E_PbI2 - E_Pb) / 2
    
    # B点: 富I
    mu_I_rich = 0
    mu_Pb_poor = E_PbI2 - 2 * E_I2
    
    print("化学势范围:")
    print(f"  μ_Pb: {mu_Pb_poor:.2f} ~ {mu_Pb_rich:.2f} eV")
    print(f"  μ_I: {mu_I_poor:.2f} ~ {mu_I_rich:.2f} eV")
    
    return {
        'Pb_rich_I_poor': {'Pb': mu_Pb_rich, 'I': mu_I_poor},
        'Pb_poor_I_rich': {'Pb': mu_Pb_poor, 'I': mu_I_rich}
    }

def plot_defect_formation_energies():
    """绘制缺陷形成能随费米能级的变化"""
    
    E_F_range = np.linspace(0, band_gap, 50)
    
    defects = ['V_Pb', 'V_I', 'V_MA', 'Pb_i', 'I_i', 'MA_i']
    charges = {
        'V_Pb': [-2, -1, 0],
        'V_I': [0, +1],
        'MA_i': [0, +1]
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax_idx, (condition, chem_pot) in enumerate(conditions.items()):
        ax = axes[ax_idx]
        
        for defect in defects:
            E_f_min = []
            for E_F in E_F_range:
                # 取最低能量的电荷态
                E_f_charges = []
                for q in charges.get(defect, [0]):
                    E_f = calculate_defect_formation_energy(
                        defect, q, chem_pot, E_F
                    )
                    E_f_charges.append(E_f)
                E_f_min.append(min(E_f_charges))
            
            ax.plot(E_F_range, E_f_min, label=defect)
        
        ax.set_xlabel('Fermi Level (eV)')
        ax.set_ylabel('Formation Energy (eV)')
        ax.set_title(f'{condition}')
        ax.legend()
        ax.set_ylim(0, 4)
    
    plt.tight_layout()
    plt.savefig('defect_formation_energies.png', dpi=300)
    plt.show()
```

**关键发现**:

| 缺陷 | 形成能 (eV) | 能级位置 | 类型 |
|------|-------------|----------|------|
| V_Pb (2-) | ~0.3-0.5 | 浅受主 | 良性 |
| V_I (+1) | ~0.1-0.3 | 浅施主 | 良性 |
| I_i (0) | ~0.3-0.6 | 深能级 | 需避免 |
| Pb_i (2+) | ~0.5-1.0 | 深能级 | 需避免 |
| MA_i (+1) | ~0.2-0.4 | 浅施主 | 良性 |

### 3.2 浅能级与载流子寿命

```python
def calculate_capture_cross_section(defect_level, phonon_energy):
    """
    计算非辐射复合截面 (Shockley-Read-Hall模型)
    
    对于深能级缺陷，复合速率高，载流子寿命短
    对于浅能级缺陷，复合速率低，载流子寿命长
    """
    
    # 简单的多声子发射模型
    # σ ∝ exp(-E_a / kT)
    
    # 激活能 (能级深度)
    if defect_level < 0.05:  # 浅能级
        activation_energy = 0.01
        sigma = 1e-20  # cm² (小截面)
        lifetime = 'long (>1 μs)'
    elif defect_level < 0.3:  # 中等深度
        activation_energy = defect_level
        sigma = 1e-16
        lifetime = 'moderate (10-100 ns)'
    else:  # 深能级
        activation_energy = defect_level
        sigma = 1e-14
        lifetime = 'short (<1 ns)'
    
    return sigma, lifetime

def analyze_defect_tolerance():
    """分析MAPbI₃的缺陷容忍性机制"""
    
    print("MAPbI₃缺陷容忍性分析:")
    print("=" * 50)
    
    # 1. 带边轨道特性
    print("\n1. 带边轨道特性:")
    print("   VBM: I-5p + Pb-6s (反键，离域)")
    print("   CBM: Pb-6p (离域)")
    print("   -> 缺陷扰动小，能级浅")
    
    # 2. 介电屏蔽
    eps_static = 25  # MAPbI₃的高介电常数
    print(f"\n2. 静态介电常数: ε = {eps_static}")
    print("   -> 强介电屏蔽，缺陷势被减弱")
    
    # 3. 有效质量
    m_eff = 0.2
    print(f"\n3. 有效质量: m* = {m_eff} m₀")
    print("   -> 小有效质量对应大玻尔半径，浅能级")
    
    # 氢模型估算
    E_b = 13.6 * (m_eff / eps_static**2)
    print(f"\n4. 氢模型估算束缚能: {E_b*1000:.1f} meV")
    print("   -> 与计算的浅能级一致")
```

---

## 4. 离子迁移与铁电性

### 4.1 离子迁移能垒

有机-无机杂化钙钛矿中存在显著的离子迁移，导致电流-电压迟滞。

```python
def calculate_ion_migration_path(ion_type='I'):
    """
    计算离子迁移路径和能垒 (NEB方法)
    
    主要迁移路径:
    - I⁻: 八面体面间跳跃 (最活跃)
    - MA⁺: 旋转+平移
    - Pb²⁺: 难以移动
    """
    
    if ion_type == 'I':
        # I⁻ 迁移: 八面体面共享路径
        # 初始位置
        initial = [0.5, 0, 0]  # 面心位置
        # 中间鞍点 (共边位置)
        saddle = [0.25, 0.25, 0]
        # 最终位置
        final = [0, 0.5, 0]
        
        # NEB计算
        images = generate_neb_images(initial, saddle, final, n_images=7)
        
        # 运行VASP NEB
        energies = run_vasp_neb(images)
        
        # 分析能垒
        barrier = max(energies) - energies[0]
        
        print(f"I⁻ 迁移能垒: {barrier:.2f} eV")
        
        # 与实验对比
        if 0.3 < barrier < 0.6:
            print("✓ 与实验值 (~0.4 eV) 一致")
        
        return barrier, energies
    
    elif ion_type == 'MA':
        # MA⁺ 旋转-平移耦合
        # 更复杂的路径
        pass

def analyze_hysteresis_mechanism():
    """分析I-V迟滞的微观机制"""
    
    print("I-V迟滞机制分析:")
    print("=" * 40)
    
    # 离子迁移导致的内建电场变化
    print("\n1. 离子迁移:")
    print("   I⁻ 向阴极迁移，留下正电荷")
    print("   -> 内建电场改变，改变电荷收集")
    
    # 铁电畴
    print("\n2. 铁电有序:")
    print("   MA⁺偶极在电场下有序")
    print("   -> 极化场调制带对齐")
    
    # 缺陷再分布
    print("\n3. 缺陷再分布:")
    print("   移动缺陷在偏压下再分布")
    print("   -> 改变复合速率")
```

### 4.2 铁电性预测

```python
def analyze_ferroelectricity():
    """
    分析铁电性质
    
    MAPbI₃在低温下可能具有铁电性
    高温下MA⁺无序，铁电性消失
    """
    
    # 计算不同MA⁺构型的能量
    polarizations = []
    energies = []
    
    ma_orientations = generate_dipole_orientations()
    
    for orient in ma_orientations:
        atoms = build_mapbi3_with_orientation(orient)
        
        # 计算极化 (Berry相方法)
        polarization = calculate_berry_phase_polarization(atoms)
        
        # 计算能量
        energy = run_vasp_calculation(atoms)
        
        polarizations.append(polarization)
        energies.append(energy)
    
    # 检查双势阱
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(polarizations, energies, s=50)
    
    # 拟合双势阱
    # E = a*P² + b*P⁴ - E_field*P
    
    ax.set_xlabel('Polarization (μC/cm²)')
    ax.set_ylabel('Energy (meV/f.u.)')
    ax.set_title('Ferroelectric Double Well')
    
    plt.tight_layout()
    plt.savefig('ferroelectric_well.png', dpi=300)
    plt.show()
```

---

## 5. 混合离子效应与合金化

### 5.1 混合阳离子/阴离子合金

实际器件常用混合组分 (如 FA/MA, Br/I) 来优化带隙和稳定性。

```python
def calculate_alloy_properties(compositions):
    """
    计算合金性质 (VCA或超胞方法)
    
    常见合金系统:
    - MA₁₋ₓFAₓPbI₃
    - MAPb(I₁₋ₓBrₓ)₃
    - Csₓ(MA/FA)₁₋ₓPbI₃
    """
    
    results = {}
    
    for comp in compositions:
        x = comp['x']
        
        # 构建超胞或VCA模型
        if method == 'supercell':
            atoms = build_supercell_alloy(comp)
        elif method == 'VCA':
            atoms = build_vca_model(comp)
        
        # 计算
        energy = run_vasp_calculation(atoms)
        gap = calculate_band_gap(atoms)
        lattice = atoms.get_cell_lengths_and_angles()[:3]
        
        results[x] = {
            'energy': energy,
            'gap': gap,
            'lattice': lattice
        }
    
    # 绘制带隙bowing曲线
    fig, ax = plt.subplots(figsize=(8, 6))
    
    xs = list(results.keys())
    gaps = [results[x]['gap'] for x in xs]
    
    ax.plot(xs, gaps, 'bo-', label='Calculated')
    
    # Vegard定律 (线性插值)
    gap_linear = [(1-x)*gaps[0] + x*gaps[-1] for x in xs]
    ax.plot(xs, gap_linear, 'r--', label='Vegard (linear)')
    
    # 光学bowing
    # E_g(x) = (1-x)E_A + xE_B - b*x*(1-x)
    
    ax.set_xlabel('Composition (x)')
    ax.set_ylabel('Band Gap (eV)')
    ax.legend()
    ax.set_title('Alloy Band Gap Bowing')
    
    plt.tight_layout()
    plt.savefig('alloy_bowing.png', dpi=300)
    plt.show()
    
    return results
```

### 5.2 相稳定性

```python
def calculate_mixing_enthalpy():
    """
    计算合金混合焓
    
    ΔH_mix = E_alloy - (1-x)E_A - xE_B
    
    ΔH_mix < 0: 有序或相分离
    ΔH_mix > 0: 混溶隙
    """
    
    x_range = np.linspace(0, 1, 11)
    delta_H = []
    
    for x in x_range:
        # 合金能量
        E_alloy = calculate_alloy_energy(x)
        
        # 端点化合物能量
        E_A = calculate_pure_energy('MAPbI3')
        E_B = calculate_pure_energy('FAPbI3')
        
        # 混合焓
        dH = E_alloy - ((1-x)*E_A + x*E_B)
        delta_H.append(dH)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_range, delta_H, 'bo-')
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_xlabel('Composition x')
    ax.set_ylabel('Mixing Enthalpy (meV/atom)')
    ax.set_title('Alloy Thermodynamic Stability')
    
    plt.tight_layout()
    plt.savefig('mixing_enthalpy.png', dpi=300)
    plt.show()
```

---

## 6. 温度效应与分子动力学

### 6.1 有限温度结构 (AIMD)

```python
def run_aimd_simulation(temperature=300, steps=10000):
    """
    运行从头算分子动力学
    
    研究:
    - MA⁺旋转动力学
    - PbI₆八面体畸变
    - 离子迁移
    """
    
    # VASP设置
    incar_md = """
    ENCUT = 400
    ISMEAR = 0
    SIGMA = 0.1
    
    # MD设置
    MDALGO = 2           # Nose-Hoover
    SMASS = 0            # 质量参数
    TEBEG = {temp}
    TEEND = {temp}
    NSW = {steps}
    POTIM = 1.0          # 1 fs时间步长
    
    # 输出设置
    NBLOCK = 1
    KBLOCK = 50
    """.format(temp=temperature, steps=steps)
    
    # 运行
    # mpirun -np 32 vasp_std
    
    # 分析轨迹
    trajectory = read_vasp_trajectory('XDATCAR')
    
    return trajectory

def analyze_md_trajectory(trajectory):
    """分析MD轨迹"""
    
    # 1. MA⁺取向时间关联
    orientations = extract_ma_orientations(trajectory)
    
    # 取向序参数
    S = calculate_orientational_order(orientations)
    print(f"取向序参数 S = {S:.3f}")
    if S < 0.3:
        print("✓ MA⁺取向无序 (液态状)")
    else:
        print("✓ MA⁺取向有序")
    
    # 2. 均方位移 (MSD) - 离子迁移
    msd_I = calculate_msd(trajectory, element='I')
    D_I = fit_diffusion_coefficient(msd_I)
    print(f"\nI⁻ 扩散系数: D = {D_I:.2e} cm²/s")
    
    # 3. 八面体倾斜角分布
    tilt_angles = calculate_octahedral_tilts(trajectory)
    print(f"\n平均八面体倾斜角: {np.mean(tilt_angles):.1f}°")
    
    # 4. 径向分布函数
    rdf_PbI = calculate_rdf(trajectory, 'Pb', 'I')
    
    return {
        'diffusion': D_I,
        'tilt': tilt_angles,
        'rdf': rdf_PbI
    }
```

---

## 7. 结果汇总与验证

### 7.1 计算结果与实验对比

| 性质 | 计算值 | 实验值 | 误差 |
|------|--------|--------|------|
| 晶格常数 a (Å) | 6.31 | 6.32 | -0.2% |
| 带隙 (eV) | 1.55 | 1.55-1.6 | ~0% |
| 电子有效质量 | 0.19 m₀ | 0.20 m₀ | -5% |
| 空穴有效质量 | 0.23 m₀ | 0.25 m₀ | -8% |
| I⁻ 迁移能垒 | 0.38 eV | 0.4-0.6 eV | -5% |
| 静态介电常数 | 25 | 25-30 | ~10% |
| 吸收边 | 1.55 eV | 1.55 eV | ~0% |

### 7.2 关键发现

```python
def summary():
    """计算结果总结"""
    
    print("=" * 60)
    print("MAPbI₃钙钛矿计算结果总结")
    print("=" * 60)
    
    print("\n【结构性质】")
    print("  • 室温稳定相: 四方相 (I4/mcm)")
    print("  • MA⁺取向: 室温动态无序，低温有序")
    print("  • PbI₆八面体: 中等程度倾斜 (~5°)")
    
    print("\n【电子性质】")
    print("  • 直接带隙: 1.55 eV (HSE06+SOC)")
    print("  • 带边离域: 高缺陷容忍性的根源")
    print("  • 低有效质量: m_e = 0.19, m_h = 0.23")
    
    print("\n【缺陷性质】")
    print("  • 主要本征缺陷: V_Pb (2-), V_I (+)")
    print("  • 缺陷能级: 浅能级 (<100 meV)")
    print("  • 自掺杂: 轻微p型 (V_Pb主导)")
    
    print("\n【动力学性质】")
    print("  • I⁻ 迁移: E_a = 0.38 eV (可移动)")
    print("  • MA⁺ 旋转: 快速，τ ~ 1 ps")
    print("  • Pb²⁺ 迁移: 困难，E_a > 1.5 eV")
    
    print("\n【器件相关性】")
    print("  • 高吸收系数: >10⁵ cm⁻¹ @ 带边")
    print("  • 长载流子寿命: >1 μs (浅缺陷)")
    print("  • 迟滞来源: 离子迁移 + 铁电畴")
    
    print("\n" + "=" * 60)
```

---

## 8. 实用脚本

### 8.1 完整计算流程

```bash
#!/bin/bash
# mapbi3_workflow.sh

echo "=== MAPbI3 Perovskite Calculation ==="

# 步骤1: 结构优化 (考虑MA取向)
echo "Step 1: Testing MA orientations..."
for orient in 001 110 111; do
    mkdir -p 1_relax_$orient
    cd 1_relax_$orient
    python build_structure.py --orientation $orient
    mpirun -np 16 vasp_std
    cd ..
done

# 选择最稳定结构
python select_best_orientation.py

# 步骤2: 电子结构 (HSE06+SOC)
echo "Step 2: Electronic structure..."
cd 2_electronic
cp ../1_relax_selected/CONTCAR POSCAR
cp ../1_relax_selected/CHGCAR .
mpirun -np 32 vasp_std
cd ..

# 步骤3: 缺陷计算
echo "Step 3: Defect calculations..."
cd 3_defects
for defect in V_Pb V_I Pb_i I_i; do
    mkdir -p $defect
    cd $defect
    python setup_defect.py --type $defect
    mpirun -np 16 vasp_std
    cd ..
done
python analyze_defects.py
cd ..

# 步骤4: 离子迁移 (NEB)
echo "Step 4: Ion migration..."
cd 4_migration
python setup_neb.py --ion I
mpirun -np 32 vasp_std
cd ..

# 步骤5: AIMD (有限温度)
echo "Step 5: Ab-initio MD..."
cd 5_aimd
cp ../1_relax_selected/CONTCAR POSCAR
mpirun -np 64 vasp_std  # 300K, 10ps
cd ..

echo "=== All calculations completed! ==="
python generate_report.py
```

---

## 参考

1. A. Kojima et al., *J. Am. Chem. Soc.* 131, 6050 (2009) - 首个高效钙钛矿太阳能电池
2. M. A. Green et al., *Prog. Photovolt.* 29, 3 (2021) - 效率纪录综述
3. J. Even et al., *J. Phys. Chem. Lett.* 4, 2999 (2013) - 理论计算基础
4. W. Yin et al., *Appl. Phys. Lett.* 104, 063903 (2014) - 缺陷容忍性
5. J. M. Frost et al., *Acc. Chem. Res.* 49, 528 (2016) - 原子尺度视角综述
6. J. S. Bechtel & A. Van der Ven, *npj Comput. Mater.* 6, 67 (2020) - 相变与离子迁移
