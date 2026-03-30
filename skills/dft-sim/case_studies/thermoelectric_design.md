# 热电材料设计 (Thermoelectric Materials)

## 背景

热电材料可实现热能与电能的直接转换，用于废热回收和固态制冷。ZT值 (品质因子) 是衡量热电性能的关键指标：

$$ZT = \frac{S^2 \sigma T}{\kappa}$$

其中 $S$ 为Seebeck系数，$\sigma$ 为电导率，$\kappa$ 为热导率，$T$ 为温度。

---

## DFT计算流程

### 1. 电子输运性质 (BoltzTraP/BoltzWann)

```python
#!/usr/bin/env python3
"""电子输运计算 - BoltzWann流程"""

import numpy as np
import matplotlib.pyplot as plt

def boltzwann_workflow():
    """BoltzWann完整计算流程"""
    
    workflow = """
    Step 1: DFT自洽计算 (pw.x)
    - 高密度k点网格 (如 16×16×16)
    - 收敛波函数
    
    Step 2: Wannier函数化 (pw2wannier90 + wannier90)
    - 投影选择 (sp³等)
    - 最大局域化
    
    Step 3: Wannier插值能带
    - 验证Wannier拟合质量
    - 检查费米面附近
    
    Step 4: BoltzWann输运计算
    - 能量网格 (DE=0.01 eV)
    - 温度范围 (100-1000 K)
    - 化学势扫描
    
    Step 5: 后处理
    - 提取 S, σ, PF
    - 计算ZT (需热导率)
    """
    
    print(workflow)

def parse_boltzwann_output(filename='boltzwann.out'):
    """解析BoltzWann输出"""
    
    # 数据存储
    temperatures = []
    chemical_potentials = []
    seebeck = []
    conductivity = []
    
    with open(filename) as f:
        for line in f:
            if 'Temperature' in line:
                T = float(line.split()[-1])
                temperatures.append(T)
            elif 'Chemical Potential' in line:
                mu = float(line.split()[-1])
                chemical_potentials.append(mu)
            elif 'Seebeck' in line:
                S = float(line.split()[-1])
                seebeck.append(S)
            elif 'Conductivity' in line:
                sigma = float(line.split()[-1])
                conductivity.append(sigma)
    
    return {
        'T': np.array(temperatures),
        'mu': np.array(chemical_potentials),
        'S': np.array(seebeck),
        'sigma': np.array(conductivity)
    }

def plot_transport_properties(data):
    """绘制输运性质随化学势变化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 提取特定温度 (如800K)
    T_target = 800
    mask = np.abs(data['T'] - T_target) < 10
    
    mu = data['mu'][mask]
    S = data['S'][mask]
    sigma = data['sigma'][mask]
    
    # Seebeck系数
    ax1 = axes[0, 0]
    ax1.plot(mu, S * 1e6, 'b-', lw=2)  # μV/K
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Chemical Potential (eV)', fontsize=11)
    ax1.set_ylabel('Seebeck S (μV/K)', fontsize=11)
    ax1.set_title(f'Seebeck Coefficient @ {T_target}K', fontsize=12)
    
    # 电导率
    ax2 = axes[0, 1]
    ax2.semilogy(mu, sigma, 'r-', lw=2)
    ax2.set_xlabel('Chemical Potential (eV)', fontsize=11)
    ax2.set_ylabel('Conductivity σ (1/Ω·m)', fontsize=11)
    ax2.set_title(f'Electrical Conductivity @ {T_target}K', fontsize=12)
    
    # 功率因子 S²σ
    pf = S**2 * sigma
    ax3 = axes[1, 0]
    ax3.plot(mu, pf * 1e6, 'g-', lw=2)  # μW/(cm·K²)
    ax3.set_xlabel('Chemical Potential (eV)', fontsize=11)
    ax3.set_ylabel('PF S²σ (μW/(cm·K²))', fontsize=11)
    ax3.set_title('Power Factor', fontsize=12)
    
    # 载流子浓度依赖
    ax4 = axes[1, 1]
    # n = ∫ DOS·f dE
    # 简化示意
    
    plt.tight_layout()
    plt.savefig('transport_properties.png', dpi=150)
    
    return fig

if __name__ == '__main__':
    boltzwann_workflow()
```

```bash
# QE + Wannier90 + BoltzWann 输入示例

# 1. pw.x输入 (自洽)
cat > scf.in << 'EOF'
&CONTROL
calculation = 'scf',
prefix = 'pbte',
outdir = './tmp',
/
&SYSTEM
ibrav = 2,          # fcc
celldm(1) = 11.2,
nat = 2, ntyp = 2,
ecutwfc = 80,
ecutrho = 800,
occupations = 'smearing',
smearing = 'mp',
degauss = 0.02,
/
&ELECTRONS
conv_thr = 1.0d-12,
/
ATOMIC_SPECIES
Pb 207.2 Pb.pbe-dn-kjpaw_psl.0.3.0.UPF
Te 127.6 Te.pbe-n-kjpaw_psl.0.3.0.UPF
ATOMIC_POSITIONS alat
Pb 0.00 0.00 0.00
Te 0.25 0.25 0.25
K_POINTS automatic
16 16 16 0 0 0
EOF

# 2. nscf计算 (更密的k点)
cat > nscf.in << 'EOF'
&CONTROL
calculation = 'nscf',
prefix = 'pbte',
outdir = './tmp',
/
&SYSTEM
... (同scf)
nbnd = 50,          # 更多能带
/
K_POINTS crystal
8000               # 显式k点或自动
EOF

# 3. pw2wannier90
cat > pw2wan.in << 'EOF'
&inputpp
prefix = 'pbte',
outdir = './tmp',
seedname = 'pbte',
write_mmn = .true.,
write_amn = .true.,
write_unk = .true.,
/
EOF

# 4. wannier90.win
cat > pbte.win << 'EOF'
num_bands = 30
num_wann = 14

begin projections
Pb: sp3
Te: sp3
end projections

dis_win_min = -8.0
dis_win_max = 6.0
dis_froz_min = -6.0
dis_froz_max = 2.0

num_iter = 1000

# BoltzWann设置
boltzwann = true
boltz_calc_also_dos = true
boltz_dos_energy_step = 0.01
boltz_mu_min = -2.0
boltz_mu_max = 2.0
boltz_mu_step = 0.05
boltz_temp_min = 300
boltz_temp_max = 1000
boltz_temp_step = 100

kmesh = 40 40 40
EOF
```

### 2. 声子与晶格热导率

```bash
# ShengBTE (迭代求解声子BTE)

# 1. 准备输入文件 CONTROL
# CONTROL
&allocations
        num_elements = 2
        num_atoms = 2
&end allocations
&crystal
        elements = "Pb Te"
        lfactor = 0.5291772
        lattice(:,1) = 0.0 0.5 0.5
        lattice(:,2) = 0.5 0.0 0.5
        lattice(:,3) = 0.5 0.5 0.0
        coordinates(:,1) = 0.0 0.0 0.0
        coordinates(:,2) = 0.25 0.25 0.25
        scell(:) = 5 5 5
&end crystal
&parameters
        T_min = 300.0
        T_max = 1000.0
        T_step = 100.0
        ngrid(:) = 20 20 20
        norientation = 0
&end parameters
&numerics
        convolution_threshold = 0.01
        smearing = 0.001
&end numerics

# 2. 运行
ShengBTE

# 3. 输出文件
# BTE.kappa_tensor  - 热导率张量
# BTE.gruneisen     - Grüneisen参数
# BTE.warnings      - 警告信息
```

```python
#!/usr/bin/env python3
"""分析ShengBTE输出"""

import numpy as np
import matplotlib.pyplot as plt

def parse_shengbte_kappa(filename='BTE.kappa_tensor'):
    """解析热导率张量"""
    
    data = np.loadtxt(filename, skiprows=1)
    
    temperature = data[:, 0]  # K
    kappa_xx = data[:, 1]     # W/(m·K)
    kappa_yy = data[:, 2]
    kappa_zz = data[:, 3]
    kappa_avg = (kappa_xx + kappa_yy + kappa_zz) / 3
    
    return temperature, kappa_avg, kappa_xx, kappa_yy, kappa_zz

def plot_thermal_conductivity(T, kappa):
    """绘制热导率随温度变化"""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(T, kappa, 'bo-', markersize=8, lw=2)
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Thermal Conductivity κ (W/(m·K))', fontsize=12)
    ax.set_title('Lattice Thermal Conductivity', fontsize=14)
    
    # 1/T趋势线
    from scipy.optimize import curve_fit
    def kappa_model(T, A, B):
        return A / T + B
    
    popt, _ = curve_fit(kappa_model, T[3:], kappa[3:])
    T_fit = np.linspace(T[3], T[-1], 100)
    ax.plot(T_fit, kappa_model(T_fit, *popt), 'r--', 
            label=f'κ ~ 1/T fit')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig('thermal_conductivity.png', dpi=150)

if __name__ == '__main__':
    T, kappa_avg, *_ = parse_shengbte_kappa()
    plot_thermal_conductivity(T, kappa_avg)
```

### 3. 载流子散射与弛豫时间

```python
#!/usr/bin/env python3
"""载流子散射计算 (EPW)"""

def epw_thermoelectric_workflow():
    """EPW热电输运计算流程"""
    
    workflow = """
    EPW完整流程:
    
    1. 电子部分 (pw.x)
       - 自洽 + 非自洽 (粗网格)
       - wannierization
       
    2. 声子部分 (ph.x + q2r + matdyn)
       - DFPT计算
       - 实空间力常数
       
    3. EPW计算
       - 电子-声子矩阵元
       - 精细网格插值
       
    4. 输运计算
       - 自能计算
       - 弛豫时间近似 (RTA)
       - 迁移率/Seebeck
    
    关键参数:
    - 电子k网格: 16×16×16 (粗) → 48×48×48 (细)
    - 声子q网格: 8×8×8 → 24×24×24
    - 温度范围: 100-1000 K
    """
    
    print(workflow)

# EPW输入示例
cat > epw.in << 'EOF'
&inputepw
prefix = 'pbte',
amass(1) = 207.2,
amass(2) = 127.6,
outdir = './tmp',
epwwrite = .true.,
epwread = .false.,
nbndsub = 14,
nbndskip = 6,
wannierize = .true.,
num_iter = 300,
proj(1) = 'Pb:l=1;l=2'
proj(2) = 'Te:l=1;l=2',
iverbosity = 2,
phselfen = .true.,
temp(1) = 300,
temp(2) = 500,
temp(3) = 800,
delta_smear = 0.01,
fermi_energy = 5.2,
nkf1 = 48, nkf2 = 48, nkf3 = 48,
nqf1 = 24, nqf2 = 24, nqf3 = 24,
dvscf_dir = './save'
band_plot = .true.,
filqf = './qpath.txt',
scattering = .true.,
int_mob = .true.,
carrier = .true.,
ncarrier = -1e20,  # 电子浓度 (cm⁻³)
iter_bte = 3,
tphases = .true.,
&end inputepw
EOF
```

---

## 案例：SnSe热电材料

### 晶体结构与相变

```python
#!/usr/bin/env python3
"""SnSe热电材料研究"""

from pymatgen.core import Structure, Lattice

def snse_structure():
    """SnSe层状正交结构 (Pnma, 低温相)"""
    
    # Pnma结构参数
    lattice = Lattice.orthorhombic(11.57, 4.15, 4.44)
    
    structure = Structure(
        lattice,
        ['Sn', 'Sn', 'Se', 'Se'],
        [
            [0.119, 0.25, 0.102],
            [0.881, 0.75, 0.898],
            [0.080, 0.25, 0.617],
            [0.920, 0.75, 0.383]
        ]
    )
    
    print("SnSe (Pnma) Structure:")
    print(f"a = {lattice.a:.3f} Å")
    print(f"b = {lattice.b:.3f} Å")
    print(f"c = {lattice.c:.3f} Å")
    
    # 高温Cmcm相
    print("\nHigh-T Cmcm phase: a ≈ 11.7, b ≈ 4.3, c ≈ 4.6 Å")
    
    return structure

def anisotropic_transport():
    """SnSe各向异性输运"""
    
    # SnSe强各向异性源于层状结构
    
    anisotropy = """
    SnSe各向异性特征:
    
    1. 晶体结构
       - 层状正交结构
       - a轴: 层内锯齿链方向
       - b轴: 层内垂直方向  
       - c轴: 层间方向
    
    2. 电子输运 (各向异性较小)
       - 多能谷贡献
       - 能带收敛 (band convergence)
       - Seebeck系数各向同性约10%
    
    3. 热输运 (高度各向异性)
       - κ_zz (层间) << κ_aa, κ_bb
       - 低κ源于强非简谐性
       - 二阶铁电相变影响
    
    4. 最优取向
       - 沿b轴: 最高ZT (记录值2.6 @ 923K)
       - 织构化提高性能
    """
    
    print(anisotropy)
    
    # 典型数值
    data = {
        'T': 923,  # K
        'S_b': 250,  # μV/K (b轴)
        'sigma_b': 250,  # S/cm
        'kappa_lat': 0.35,  # W/(m·K)
        'kappa_elec': 0.5,  # W/(m·K)
    }
    
    PF = (data['S_b'] * 1e-6)**2 * data['sigma_b'] * 100  # W/(m·K²)
    ZT = PF * data['T'] / (data['kappa_lat'] + data['kappa_elec'])
    
    print(f"\nTypical values @ {data['T']}K (b-axis):")
    print(f"Power factor: {PF*1e3:.2f} mW/(m·K²)")
    print(f"ZT value: {ZT:.2f}")

def band_convergence():
    """能带收敛工程"""
    
    explanation = """
    SnSe能带收敛策略:
    
    1. 问题: 多能谷不简并 → 态密度有效质量小
    
    2. 解决: 通过温度/掺杂/应变使能谷收敛
    
    3. DFT计算:
       - 计算不同k点价带顶能量
       - 跟踪随温度的变化
       - 预测最优掺杂浓度
    
    4. 结果: 
       - 823K时L和Z能谷收敛
       - 有效质量增加 → Seebeck提升
       - 功率因子优化
    """
    
    print(explanation)

if __name__ == '__main__':
    snse_structure()
    anisotropic_transport()
    band_convergence()
```

### Na掺杂优化

```python
def na_doping_optimization():
    """Na掺杂SnSe优化"""
    
    print("="*60)
    print("Na-doped SnSe Optimization")
    print("="*60)
    
    # DFT计算方案
    calc_setup = """
    计算方法:
    
    1. 超胞构建
       - 2×2×2超胞 (64原子)
       - 或 3×3×2 (72原子)
       
    2. 掺杂构型
       - 替换Sn位
       - 测试不同掺杂浓度
       - 考虑有序/无序
       
    3. 关键计算
       - 形成能 (确定溶解度)
       - 载流子浓度
       - 能带结构变化
       
    4. 输运计算
       - 掺杂后电子结构
       - 有效质量
       - 散射时间估计
    """
    
    print(calc_setup)
    
    # 实验对比
    results = {
        'undoped': {'p': 1e17, 'S': 550, 'sigma': 10, 'ZT': 0.5},
        'Na_0.02': {'p': 5e19, 'S': 300, 'sigma': 150, 'ZT': 1.5},
        'Na_0.05': {'p': 2e20, 'S': 200, 'sigma': 400, 'ZT': 2.2},
    }
    
    print("\nCarrier Concentration Optimization:")
    print(f"{'Doping':<15}{'p(cm⁻³)':<15}{'S(μV/K)':<12}{'σ(S/cm)':<12}{'ZT':<8}")
    print("-"*60)
    for name, data in results.items():
        print(f"{name:<15}{data['p']:.0e}{'':<5}{data['S']:<12}{data['sigma']:<12}{data['ZT']:<8.2f}")
```

---

## 案例：Half-Heusler合金

```python
#!/usr/bin/env python3
"""Half-Heusler热电合金研究"""

def half_heusler_overview():
    """Half-Heusler合金概述"""
    
    overview = """
    Half-Heusler合金 (XYZ):
    
    结构: 立方Clb相 (空间群 F-43m)
    
    价电子计数规则:
    - 18电子体系: 半导体/半金属
    - 17电子: n型
    - 19电子: p型
    
    典型体系:
    - p型: NbFeSb, ZrCoSb, TiCoSb
    - n型: ZrNiSn, TiNiSn, NbCoSn
    
    优势:
    - 机械性能好
    - 热稳定性高
    - 可调控性强
    
    挑战:
    - 晶格热导率偏高
    - 需要纳米结构化
    """
    
    print(overview)

def nbesb_dft_study():
    """NbFeSb DFT研究示例"""
    
    print("="*60)
    print("NbFeSb Half-Heusler Study")
    print("="*60)
    
    # VASP输入
    vasp_input = """
    INCAR设置:
    
    SYSTEM = NbFeSb Half-Heusler
    ENCUT = 500
    ISMEAR = -5  # 精确DOS
    
    # 能带计算
    LORBIT = 11  # 投影DOS
    NEDOS = 5001
    
    # 杂化泛函 (带隙修正)
    LHFCALC = .TRUE.
    HFSCREEN = 0.2  # HSE06
    """
    
    print(vasp_input)
    
    # 结果分析
    properties = {
        'lattice_constant': 5.94,  # Å
        'band_gap': 0.45,  # eV (HSE06)
        'S_900K': 250,  # μV/K
        'sigma_900K': 800,  # S/cm
        'kappa_900K': 4.5,  # W/(m·K)
    }
    
    ZT = (properties['S_900K']*1e-6)**2 * properties['sigma_900K']*100 * 900 / properties['kappa_900K']
    
    print(f"\nProperties @ 900K:")
    print(f"Band gap: {properties['band_gap']:.2f} eV")
    print(f"ZT value: {ZT:.2f}")

def hafnium_doping():
    """Hf掺杂优化 (能带收敛)"""
    
    print("\nHf-doping for Band Convergence:")
    
    mechanism = """
    NbFeSb-Hf优化策略:
    
    1. 能带结构特征
       - VBM: Γ点 (轻带)
       - 次高 valence: Σ点 (重带)
       - 能量差 ~0.1 eV
       
    2. Hf掺杂效应
       - Hf替代Nb
       - 增加轨道重叠
       - 轻/重带能量靠近
       
    3. DFT预测
       - 计算不同Hf含量能带
       - 追踪轻重带能量差
       - 优化掺杂浓度
       
    4. 实验验证
       - 最优x=0.1-0.2
       - Seebeck提升30%
       - ZT ~1.5 @ 1200K
    """
    
    print(mechanism)

if __name__ == '__main__':
    half_heusler_overview()
    nbesb_dft_study()
    hafnium_doping()
```

---

## 高通量筛选策略

```python
#!/usr/bin/env python3
"""热电材料高通量筛选"""

def high_throughput_screening():
    """高通量筛选流程"""
    
    workflow = """
    高通量热电筛选:
    
    第1步: 结构获取
    - Materials Project API
    - ICSD晶体结构
    - 结构原型枚举
    
    第2步: 快速计算
    - GGA-DFT优化
    - 静态能带计算
    - 声子稳定性 (或ML预测)
    
    第3步: 描述符筛选
    - 带隙: 0.1-1.0 eV
    - 有效质量: 0.5-5 m0
    - 德拜温度: 低优先
    - Grüneisen参数: 大优先
    
    第4步: 精细计算
    - 电输运 (BoltzTraP)
    - 声子 (DFPT/ShengBTE)
    - 缺陷 (formation energy)
    
    第5步: 实验验证
    """
    
    print(workflow)
    
    # 机器学习描述符
    descriptors = {
        'composition': ['mean_atomic_mass', 'std_pauling_en', 'avg_ion_rad'],
        'structure': ['density', ' Packing fraction', 'coordination'],
        'electronic': ['band_gap', 'effective_mass', 'DOS_at_EF'],
        'phonon': ['debye_temp', 'gruneisen', 'acoustic_ph_freq']
    }
    
    print("\nKey Descriptors:")
    for category, desc_list in descriptors.items():
        print(f"  {category}: {', '.join(desc_list)}")

def machine_learning_zt():
    """机器学习预测ZT"""
    
    ml_approach = """
    ML for ZT prediction:
    
    模型类型:
    1. 结构无关 (仅组成)
      - 快速筛选
      - 准确性较低
      
    2. 结构相关
      - CGCNN, MEGNet
      - 需要晶体结构
      - 更准确
    
    3. 多保真度
      - 结合DFT+实验数据
      - 主动学习
    
    推荐软件:
    - matminer (特征提取)
    - CGCNN (图神经网络)
    - AFLOW-ML (在线服务)
    """
    
    print(ml_approach)

if __name__ == '__main__':
    high_throughput_screening()
    machine_learning_zt()
```

---

## 参考资源

- 综述: Snyder, Toberer, "Complex Thermoelectric Materials", Nature Mater. 2008
- 综述: Zhu et al., "Computational Understanding of Thermoelectrics", Chem. Mater. 2021
- BoltzTraP: Madsen, Singh, Comput. Phys. Commun. 2006
- ShengBTE: Li et al., Comput. Phys. Commun. 2014
- EPW: Ponce et al., Comput. Phys. Commun. 2016
- 数据库: AFLOW (aflowlib.org), Materials Project

---

*案例作者: DFT-Sim Team*
*最后更新: 2026-03-08*
