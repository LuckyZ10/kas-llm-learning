# 高级分析方法: Wannier函数、Berry相与非线性响应

## 概述

本模块介绍DFT计算中的高级分析方法，包括Wannier函数插值、Berry相计算、拓扑不变量以及非线性光学响应。这些方法对于理解材料的电子拓扑、输运和光学性质至关重要。

---

## 1. Wannier函数

### 1.1 理论基础

Wannier函数是Bloch函数的傅里叶变换，提供实空间局域化的轨道描述:

$$|R n\rangle = \frac{V}{(2\pi)^3} \int_{BZ} dk e^{-ik\cdot R} |\psi_{nk}\rangle$$

**应用**:
- 能带插值 (比DFT精细100倍k点)
- 电子输运 (BoltzWann)
- 电声耦合 (EPW)
- 紧束缚模型构建

### 1.2 Wannier90计算流程

```bash
# 1. 准备输入文件 case.win
cat > si.win << EOF
num_wann = 8
num_iter = 200

# 初始投影
begin projections
Si: sp3
end projections

# k点网格
mp_grid = 10 10 10

# 输出选项
bands_plot = true
fermi_surface_plot = true
EOF

# 2. VASP计算 (LWANNIER90 = .TRUE.)
# 需要设置 LWANNIER90 = .TRUE. 和 WANNIER90_WIN = "si.win"
mpirun -np 16 vasp_std

# 3. 运行Wannier90
wannier90.x si

# 4. 后处理 (能带插值等)
wannier90.x si -band
postw90.x si
```

**VASP输入关键参数**:
```
LWANNIER90 = .TRUE.
WANNIER90_WIN = "si.win"
NBANDS = 12           # 包含足够空带
```

### 1.3 Python接口: PythTB

```python
import numpy as np
import matplotlib.pyplot as plt
from pythtb import tb_model, wf_array

def build_tb_from_wannier(wannier_dat):
    """
    从Wannier90输出构建紧束缚模型
    """
    
    # 读取Wannier数据
    hoppings, lat, orb = read_wannier90(wannier_dat)
    
    # 构建PythTB模型
    my_model = tb_model(3, 3, lat, orb)
    
    # 添加跃迁
    for R, hop in hoppings.items():
        for i in range(len(orb)):
            for j in range(len(orb)):
                if abs(hop[i, j]) > 1e-6:
                    my_model.set_hop(hop[i, j], i, j, R)
    
    return my_model

def calculate_wannier_bands(tb_model, k_path):
    """使用紧束缚模型计算能带"""
    
    # 求解
    evals = tb_model.solve_all(k_path)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for n in range(tb_model.get_num_orbitals()):
        ax.plot(k_path, evals[n], 'b-', linewidth=1)
    
    ax.set_xlabel('k-path')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Wannier Interpolated Bands')
    
    plt.tight_layout()
    plt.savefig('wannier_bands.png', dpi=300)
    plt.show()
    
    return evals

# 计算态密度
def calculate_dos_wannier(tb_model, nk=(50, 50, 50), E_range=(-5, 5)):
    """
    使用Wannier函数计算精细态密度
    
    优势: 可用超密k点网格 (1000×1000×1000)
    """
    
    from pythtb import dos_tetrahedron
    
    # 生成k点网格
    k_grid = []
    for i in range(nk[0]):
        for j in range(nk[1]):
            for k in range(nk[2]):
                k_vec = [i/nk[0], j/nk[1], k/nk[2]]
                k_grid.append(k_vec)
    
    # 计算本征值
    energies = tb_model.solve_all(k_grid)
    
    # 四面体法计算DOS
    dos, energy_bins = dos_tetrahedron(
        energies.flatten(),
        E_range[0], E_range[1],
        1000  # 能量格点数
    )
    
    # 绘制
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(energy_bins, 0, dos, alpha=0.5)
    ax.plot(energy_bins, dos, 'b-', linewidth=1)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('DOS')
    ax.set_title(f'DOS from Wannier (nk={nk[0]}³)')
    
    plt.tight_layout()
    plt.savefig('wannier_dos.png', dpi=300)
    plt.show()
    
    return dos, energy_bins
```

### 1.4 能带插值对比

```python
def compare_dft_vs_wannier(dft_bands, wannier_bands, k_path):
    """
    对比DFT和Wannier插值能带
    
    验证Wannier插值精度
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # DFT能带
    ax = axes[0]
    for band in dft_bands:
        ax.plot(k_path, band, 'ro', markersize=3, label='DFT')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('DFT Bands (coarse k-grid)')
    ax.legend()
    
    # Wannier插值
    ax = axes[1]
    for band in dft_bands:
        ax.plot(k_path, band, 'ro', markersize=3, alpha=0.5)
    for band in wannier_bands:
        ax.plot(k_path, band, 'b-', linewidth=1.5, label='Wannier')
    ax.set_xlabel('k-path')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Wannier Interpolation (fine k-grid)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300)
    plt.show()
    
    # 计算误差
    errors = []
    for dft_band, wan_band in zip(dft_bands, wannier_bands):
        error = np.mean(np.abs(dft_band - wan_band))
        errors.append(error)
    
    print(f"平均插值误差: {np.mean(errors):.4f} eV")
    print(f"最大误差: {np.max(errors):.4f} eV")
```

---

## 2. Berry相与拓扑性质

### 2.1 Berry相理论基础

Berry相描述波函数在参数空间演化时积累的相位:

$$\gamma_n = \oint_C \langle u_{nk} | i \nabla_k | u_{nk} \rangle \cdot dk$$

**物理应用**:
- 极化 (现代极化理论)
- 拓扑不变量 (Chern数, Z₂)
- 反常霍尔电导

### 2.2 极化计算

```python
def calculate_polarization(wannier90_output):
    """
    计算电子极化 (Berry相方法)
    
    现代极化理论: P = P_ion + P_electron
    """
    
    from wannierberri import System, calculators
    
    # 加载Wannier函数
    system = System(wannier90_output)
    
    # 计算极化
    calculators_polarization = {
        'polarization': calculators.Polarization()
    }
    
    # 计算
    result = system.run(
        grid=20,  # k点网格
        calculators=calculators_polarization
    )
    
    # 提取结果
    P_electron = result['polarization']  # e·Å
    
    # 离子贡献
    P_ion = calculate_ionic_polarization(system.structure)
    
    P_total = P_ion + P_electron
    
    print(f"电子极化: {P_electron:.4f} e·Å")
    print(f"离子极化: {P_ion:.4f} e·Å")
    print(f"总极化: {P_total:.4f} e·Å")
    print(f"极化电荷: {P_total/system.volume:.4f} C/m²")
    
    return P_total
```

### 2.3 Z₂拓扑不变量 (Wannier方法)

```python
def calculate_z2_wannier(wannier90_output):
    """
    使用Wannier函数计算Z₂拓扑不变量
    
    方法: Wannier中心演化 (WCC)
    """
    
    import z2pack
    
    # 从Wannier构建系统
    system = z2pack.fp.System(
        wannier90_output['hr_file'],
        kpt_fct=[z2pack.fp.kpoint.wannier90],
        kpt_path=['wannier90.win']
    )
    
    # 定义表面 (对于3D拓扑绝缘体)
    # 计算(001)表面的Wilson loop
    result = z2pack.surface.run(
        system=system,
        surface=lambda s, t: [t, s, 0],  # kz = 0平面
        num_lines=11,
        pos_tol=1e-2,
        gap_tol=0.01
    )
    
    # 计算Z₂
    z2 = z2pack.invariant.z2(result)
    
    print(f"Z₂不变量 = {z2}")
    
    if z2 == 1:
        print("✓ 强拓扑绝缘体")
    else:
        print("✗ 平凡绝缘体")
    
    # 可视化Wannier中心
    fig, ax = plt.subplots(figsize=(8, 6))
    z2pack.plot.wcc(result, axis=ax)
    ax.set_xlabel(r'$k_y$')
    ax.set_ylabel(r'$\bar{x}$ (WCC)')
    ax.set_title('Wilson Loop / Wannier Charge Centers')
    
    plt.tight_layout()
    plt.savefig('wcc_plot.png', dpi=300)
    plt.show()
    
    return z2, result
```

### 2.4 Chern数计算 (2D体系)

```python
def calculate_chern_number(wannier_files):
    """
    计算Chern数
    
    C = (1/2π) ∫_BZ F_xy(k) d²k
    
    F_xy: Berry曲率
    """
    
    from wannierberri import System, calculators, run
    
    system = System(wannier_files)
    
    # 计算Berry曲率
    calculators_chern = {
        'berry_curvature': calculators.BerryCurvature(),
        'chern_number': calculators.ChernNumber()
    }
    
    result = run(system, grid=100, calculators=calculators_chern)
    
    chern = result['chern_number']
    
    print(f"Chern数 C = {chern}")
    
    if abs(chern) > 0.5:
        print(f"✓ 量子反常霍尔效应 (C = {int(round(chern))})")
    
    # 绘制Berry曲率
    berry_curvature = result['berry_curvature']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(berry_curvature, cmap='RdBu', origin='lower')
    plt.colorbar(im, ax=ax, label='Berry Curvature')
    ax.set_title(f'Berry Curvature (Chern = {chern:.1f})')
    
    plt.savefig('berry_curvature.png', dpi=300)
    plt.show()
    
    return chern
```

---

## 3. 非线性响应

### 3.1 非线性光学响应

```python
def calculate_shg_response(wannier_files, omega_range=(0, 10)):
    """
    计算二次谐波产生 (SHG) 响应
    
    χ⁽²⁾(-2ω; ω, ω)
    """
    
    from wannierberri import System, calculators
    
    system = System(wannier_files)
    
    # 计算SHG张量
    calculators_shg = {
        'shg': calculators.SHG(
            omega=omega_range,
            smearing=0.1
        )
    }
    
    result = run(system, grid=50, calculators=calculators_shg)
    
    # 提取SHG张量分量
    shg_tensor = result['shg']
    
    # 典型分量: χ⁽²⁾_xyz
    omega = np.linspace(omega_range[0], omega_range[1], 100)
    chi_xxx = shg_tensor[:, 0, 0, 0]  # χ_xxx
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(omega, np.abs(chi_xxx), 'b-', label='|χ⁽²⁾_xxx|')
    ax.set_xlabel('Photon Energy (eV)')
    ax.set_ylabel('|χ⁽²⁾| (pm/V)')
    ax.set_title('Second Harmonic Generation Response')
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('shg_response.png', dpi=300)
    plt.show()
    
    return shg_tensor
```

### 3.2 非线性霍尔效应

```python
def calculate_nonlinear_hall(wannier_files):
    """
    计算非线性霍尔电导
    
    σ_{ijk} = ∂²J_i / ∂E_j ∂E_k
    """
    
    from wannierberri import System, calculators
    
    system = System(wannier_files)
    
    # 计算非线性霍尔响应
    calculators_nlh = {
        'nonlinear_hall': calculators.NonlinearHall()
    }
    
    result = run(system, grid=80, calculators=calculators_nlh)
    
    sigma_nlh = result['nonlinear_hall']
    
    print("非线性霍尔电导张量:")
    print(sigma_nlh)
    
    return sigma_nlh
```

---

## 4. 输运性质

### 4.1 BoltzWann输运计算

```python
def calculate_transport_wannier(wannier_files, temperatures=[300], doping=[0]):
    """
    使用Boltzmann方程计算输运系数
    
    基于Wannier函数插值
    """
    
    from wannierberri import System, calculators
    
    system = System(wannier_files)
    
    # 设置计算
    calculators_transport = {
        'conductivity': calculators.Conductivity(
            temperatures=temperatures,
            doping=doping,
            mu=0.1  # 化学势扫描范围
        ),
        'seebeck': calculators.Seebeck(
            temperatures=temperatures,
            doping=doping
        ),
        'thermal_conductivity': calculators.ThermalConductivity(
            temperatures=temperatures
        )
    }
    
    result = run(system, grid=100, calculators=calculators_transport)
    
    # 提取结果
    conductivity = result['conductivity']
    seebeck = result['seebeck']
    
    # 绘制
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 电导率
    ax = axes[0]
    for T in temperatures:
        ax.plot(doping, conductivity[T], 'o-', label=f'{T}K')
    ax.set_xlabel('Doping (cm⁻³)')
    ax.set_ylabel('Conductivity (S/cm)')
    ax.set_title('Electrical Conductivity')
    ax.legend()
    ax.set_yscale('log')
    
    # Seebeck系数
    ax = axes[1]
    for T in temperatures:
        ax.plot(doping, seebeck[T] * 1e6, 'o-', label=f'{T}K')
    ax.set_xlabel('Doping (cm⁻³)')
    ax.set_ylabel('Seebeck Coefficient (μV/K)')
    ax.set_title('Thermopower')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('transport_properties.png', dpi=300)
    plt.show()
    
    return result
```

### 4.2 电声耦合 (EPW)

```python
def calculate_elph_coupling(epw_output):
    """
    计算电声耦合常数
    
    使用EPW (Electron-Phonon Wannier)
    """
    
    # 读取EPW结果
    lambda_epw, omega_log, mu_star = parse_epw_output(epw_output)
    
    # McMillan公式估算Tc
    Tc = (omega_log / 1.2) * np.exp(
        -1.04 * (1 + lambda_epw) / 
        (lambda_epw - mu_star * (1 + 0.62 * lambda_epw))
    )
    
    print(f"电声耦合常数 λ = {lambda_epw:.3f}")
    print(f"对数平均频率 ω_log = {omega_log:.1f} K")
    print(f"库仑赝势 μ* = {mu_star:.3f}")
    print(f"估算超导温度 Tc = {Tc:.1f} K")
    
    return {
        'lambda': lambda_epw,
        'omega_log': omega_log,
        'mu_star': mu_star,
        'Tc': Tc
    }
```

---

## 5. 软件工具

### 5.1 主要软件

| 软件 | 功能 | 接口 |
|------|------|------|
| Wannier90 | Wannier函数 | 独立/DFT接口 |
| wannierberri | Berry相/输运 | Python |
| PythTB | 紧束缚模型 | Python |
| Z2Pack | 拓扑不变量 | Python |
| EPW | 电声耦合 | QE接口 |
| BoltzTraP | 输运系数 | 独立 |

### 5.2 安装

```bash
# Wannier90
wget http://www.wannier.org/code/wannier90-3.1.0.tar.gz
tar -xzf wannier90-3.1.0.tar.gz
cd wannier90-3.1.0
make

# wannierberri
pip install wannierberri

# Z2Pack
pip install z2pack

# PythTB
pip install pythtb
```

---

## 6. 完整工作流示例

### 6.1 拓扑绝缘体完整分析

```python
def topological_insulator_analysis(vasp_output, wannier_input):
    """
    拓扑绝缘体的完整分析流程
    """
    
    print("=" * 60)
    print("拓扑绝缘体分析工作流")
    print("=" * 60)
    
    # 1. Wannier函数化
    print("\n1. 构建Wannier函数...")
    run_wannier90(wannier_input)
    
    # 2. 能带插值
    print("\n2. 能带插值...")
    tb_model = build_tb_from_wannier('wannier90_hr.dat')
    wannier_bands = calculate_wannier_bands(tb_model, k_path)
    
    # 3. Z₂计算
    print("\n3. 计算Z₂拓扑不变量...")
    z2, wcc_data = calculate_z2_wannier('wannier90_output')
    
    # 4. 表面态 (slab模型)
    if z2 == 1:
        print("\n4. 确认表面态...")
        surface_bands = calculate_surface_states(vasp_output, thickness=6)
    
    # 5. Berry曲率
    print("\n5. 计算Berry曲率...")
    berry_data = calculate_berry_curvature(tb_model)
    
    # 生成报告
    report = f"""
# 拓扑分析结果

## 拓扑分类
- Z₂不变量: ({z2};000)
- 类型: {'强拓扑绝缘体' if z2 == 1 else '平凡绝缘体'}

## 电子结构
- 带隙: {band_gap:.2f} eV
- 狄拉克速度: {dirac_velocity:.2f} × 10⁵ m/s

## 结论
{'✓ 确认为拓扑绝缘体，存在表面狄拉克锥' if z2 == 1 else '✗ 平凡绝缘体'}
"""
    
    with open('TOPOLOGY_REPORT.md', 'w') as f:
        f.write(report)
    
    print("\n分析完成! 报告已生成: TOPOLOGY_REPORT.md")
    
    return z2, wannier_bands
```

---

## 参考

1. N. Marzari et al., *Rev. Mod. Phys.* 84, 1419 (2012) - Wannier函数综述
2. D. Vanderbilt, *Berry Phases in Electronic Structure Theory* (2018) - Berry相教材
3. S. P. Ong & A. Jain, *Comput. Mater. Sci.* 97, 209 (2015) - PythTB
4. D. Gresch et al., *Phys. Rev. B* 95, 075146 (2017) - Z2Pack
5. S. Tsirkin et al., *npj Comput. Mater.* 7, 156 (2021) - wannierberri
6. S. Ponce et al., *Comput. Phys. Commun.* 209, 116 (2016) - EPW
