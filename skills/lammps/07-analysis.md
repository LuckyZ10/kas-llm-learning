# 07. 分析工具与后处理

> LAMMPS模拟数据分析、可视化和高级后处理方法

---

## 目录
- [内置分析命令](#内置分析命令)
- [径向分布函数(RDF)](#径向分布函数rdf)
- [均方位移(MSD)与扩散系数](#均方位移msd与扩散系数)
- [自相关函数](#自相关函数)
- [自由能计算](#自由能计算)
- [结构分析](#结构分析)
- [可视化工具](#可视化工具)
- [Python后处理](#python后处理)

---

## 内置分析命令

### 1. 基础计算 (compute)

```lammps
# 温度相关
compute my_temp all temp          # 温度
compute my_temp_com all temp/com  # 去除质心运动的温度
compute my_ke all ke              # 总动能

# 压强相关
compute my_press all pressure thermo_temp   # 压强张量
compute my_stress all stress/atom virial    # 每个原子的应力

# 能量相关
compute pe_atom all pe/atom       # 每个原子的势能
compute ke_atom all ke/atom       # 每个原子的动能
compute cna all cna/atom 3.5      # 共近邻分析

# 几何相关
compute com all com               # 质心
compute gyr all gyration          # 回转半径
compute msd all msd               # 均方位移
compute rdf all rdf 100 1 1 1 2 2 2  # RDF
```

### 2. 分块分析 (chunk)

```lammps
# 1D分块
compute 1d_chunk all chunk/atom bin/1d x lower 0.1
fix 1 all ave/chunk 100 10 1000 1d_chunk density/mass file density_profile.dat

# 2D分块
compute 2d_chunk all chunk/atom bin/2d x lower 0.1 y lower 0.1

# 球形壳层
compute shell_chunk all chunk/atom bin/sphere 0.0 0.0 0.0 0.0 5.0 0.5
```

### 3. 邻居分析

```lammps
# 配位数
coord/atom cutoff 3.5 group all

# 键长分布
compute bonds all property/local batom1 batom2 btype
```

---

## 径向分布函数(RDF)

### 1. LAMMPS中计算RDF

```lammps
# 方法1: 使用compute rdf
compute my_rdf all rdf 100 1 1 1 2 2 2
c_ my_rdf[1] = bin距离
c_ my_rdf[2] = 类型1-1的g(r)
c_ my_rdf[3] = 类型1-2的g(r)
c_ my_rdf[4] = 类型2-2的g(r)

# 时间平均
fix 1 all ave/time 100 5 1000 c_my_rdf[*] file rdf.dat mode vector

# 完整示例
units metal
atom_style atomic
read_data cu.data

pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

compute cu_rdf all rdf 200 1 1  # 200 bins, type 1-1
fix 1 all ave/time 100 10 10000 c_cu_rdf[*] file rdf.dat mode vector

thermo 1000
run 100000
```

### 2. Python后处理RDF

```python
# rdf_analysis.py
import numpy as np
import matplotlib.pyplot as plt

def read_lammps_rdf(filename):
    """读取LAMMPS RDF输出"""
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 查找数据块
    for i, line in enumerate(lines):
        if line.startswith('# Row'):
            # 读取数据
            data_start = i + 1
            break
    
    # 解析数据
    distances = []
    g_r = []
    for line in lines[data_start:]:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            distances.append(float(parts[1]))
            g_r.append(float(parts[2]))
    
    return np.array(distances), np.array(g_r)

def compute_coordination_number(r, g_r, rho, r_max):
    """计算配位数"""
    # CN = 4πρ ∫_0^rmax r²g(r)dr
    dr = r[1] - r[0]
    integrand = 4 * np.pi * rho * r**2 * g_r
    cn = np.trapz(integrand[r <= r_max], r[r <= r_max])
    return cn

def compute_structure_factor(r, g_r, rho, q_max=20, n_q=500):
    """计算结构因子 S(q)"""
    q = np.linspace(0.1, q_max, n_q)
    S = np.zeros_like(q)
    
    dr = r[1] - r[0]
    for i, qi in enumerate(q):
        integrand = r * (g_r - 1) * np.sin(qi * r)
        S[i] = 1 + 4 * np.pi * rho / qi * np.trapz(integrand, r)
    
    return q, S

# 使用示例
r, g_r = read_lammps_rdf('rdf.dat')

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(r, g_r, 'b-', linewidth=2)
axes[0].set_xlabel('r (Å)')
axes[0].set_ylabel('g(r)')
axes[0].set_title('Radial Distribution Function')
axes[0].axhline(y=1, color='r', linestyle='--', alpha=0.5)
axes[0].grid(True)

# 计算配位数
rho = 0.085  # 原子数密度 (atoms/Å³)
for r_max in [3.0, 4.0, 5.0]:
    cn = compute_coordination_number(r, g_r, rho, r_max)
    print(f"Coordination number (r < {r_max} Å): {cn:.2f}")

# 计算S(q)
q, S = compute_structure_factor(r, g_r, rho)
axes[1].plot(q, S, 'r-', linewidth=2)
axes[1].set_xlabel('q (Å⁻¹)')
axes[1].set_ylabel('S(q)')
axes[1].set_title('Structure Factor')
axes[1].axhline(y=1, color='k', linestyle='--', alpha=0.5)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('rdf_analysis.png', dpi=300)
```

---

## 均方位移(MSD)与扩散系数

### 1. LAMMPS中计算MSD

```lammps
# 标准MSD
compute my_msd all msd
# c_my_msd[1] = 总MSD
# c_my_msd[2] = x方向
# c_my_msd[3] = y方向
# c_my_msd[4] = z方向

# 输出
fix 1 all ave/time 10 100 1000 c_my_msd file msd.dat

# 特定组
compute water_msd water msd
fix 2 water ave/time 10 100 1000 c_water_msd file water_msd.dat

# 完整示例
units real
atom_style full
read_data water.data

group water type 1 2

compute water_msd water msd
fix 1 water ave/time 10 100 1000 c_water_msd[*] file msd.dat mode vector

dump 1 water custom 1000 water.dump id type x y z

timestep 1.0
run 100000
```

### 2. 扩散系数分析

```python
# diffusion_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def read_msd(filename):
    """读取MSD数据"""
    data = np.loadtxt(filename, comments='#')
    timestep = data[:, 0]
    msd = data[:, 1]  # 总MSD
    msd_x = data[:, 2]
    msd_y = data[:, 3]
    msd_z = data[:, 4]
    return timestep, msd, msd_x, msd_y, msd_z

def compute_diffusion_coefficient(timestep, msd, dt, dim=3):
    """
    从MSD计算扩散系数
    
    MSD = 2 * dim * D * t
    
    Parameters:
    -----------
    timestep : array
        时间步数组
    msd : array
        MSD数组 (Å²)
    dt : float
        时间步长 (fs)
    dim : int
        维度 (1, 2, or 3)
    
    Returns:
    --------
    D : float
        扩散系数 (cm²/s)
    r_squared : float
        R²拟合优度
    """
    # 转换为实际时间 (ps)
    time = timestep * dt / 1000.0  # fs to ps
    
    # 线性拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(time, msd)
    
    # D = slope / (2 * dim)
    D = slope / (2 * dim)  # Å²/ps
    
    # 转换为cm²/s
    D_cm2s = D * 1e-16 / 1e-12  # Å²/ps to cm²/s
    
    return D_cm2s, r_value**2, slope, intercept

def analyze_diffusion_by_region(filename, dt, n_regions=5):
    """分析不同区域的扩散系数"""
    # 按z坐标分区域
    pass

def einstein_relation_check(timestep, msd, dt, dim=3):
    """验证爱因斯坦关系"""
    time = timestep * dt / 1000.0
    
    # 理论上 MSD = 2*dim*D*t
    D, r2, slope, intercept = compute_diffusion_coefficient(timestep, msd, dt, dim)
    
    # 检查线性度
    # 前25%用于平衡期，后50%用于拟合
    n = len(timestep)
    start_idx = n // 4
    end_idx = n * 3 // 4
    
    slope_fit, _, r2_fit, _, _ = stats.linregress(
        time[start_idx:end_idx], 
        msd[start_idx:end_idx]
    )
    
    return {
        'D': D,
        'r2_full': r2,
        'r2_fit': r2_fit,
        'slope': slope_fit
    }

# 使用示例
timestep, msd, msd_x, msd_y, msd_z = read_msd('msd.dat')

# 计算总扩散系数
D, r2, slope, intercept = compute_diffusion_coefficient(timestep, msd, dt=1.0, dim=3)
print(f"Diffusion Coefficient: {D:.2e} cm²/s")
print(f"R² = {r2:.4f}")

# 验证爱因斯坦关系
results = einstein_relation_check(timestep, msd, dt=1.0)
print(f"\nFit region R² = {results['r2_fit']:.4f}")

# 绘图
time = timestep * 1.0 / 1000.0  # ps

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# MSD vs time
axes[0].plot(time, msd, 'b-', label='Total MSD', linewidth=2)
axes[0].plot(time, msd_x, 'r--', alpha=0.7, label='MSD_x')
axes[0].plot(time, msd_y, 'g--', alpha=0.7, label='MSD_y')
axes[0].plot(time, msd_z, 'm--', alpha=0.7, label='MSD_z')
axes[0].set_xlabel('Time (ps)')
axes[0].set_ylabel('MSD (Å²)')
axes[0].set_title('Mean Square Displacement')
axes[0].legend()
axes[0].grid(True)

# 添加拟合线
fit_line = slope * time + intercept
axes[0].plot(time, fit_line, 'k:', linewidth=2, label=f'Fit: D={D:.2e} cm²/s')

# 对数坐标检查
axes[1].loglog(time, msd, 'b-', linewidth=2)
axes[1].set_xlabel('Time (ps)')
axes[1].set_ylabel('MSD (Å²)')
axes[1].set_title('MSD (Log-Log)')
axes[1].grid(True)

# 理想扩散斜率=1
axes[1].plot(time, time, 'r--', alpha=0.5, label='Slope = 1 (Diffusive)')
axes[1].legend()

plt.tight_layout()
plt.savefig('diffusion_analysis.png', dpi=300)
```

---

## 自相关函数

### 1. 速度自相关函数(VACF)

```lammps
# 计算VACF
compute vacf all vacf
# c_vacf[1] = VACF
# c_vacf[2] = x分量
# c_vacf[3] = y分量
# c_vacf[4] = z分量

# 时间相关
fix 1 all ave/correlate 10 100 1000 c_vacf[1] type auto file vacf.dat

# 完整示例
units real
atom_style atomic

compute vacf all vacf
fix 1 all ave/correlate 10 100 1000 &
    c_vacf[1] c_vacf[2] c_vacf[3] c_vacf[4] &
    type auto file vacf.dat ave running

timestep 1.0
run 100000
```

### 2. 应力自相关函数 (粘度计算)

```lammps
# Green-Kubo粘度
compute pressure all pressure thermo_temp
fix 1 all ave/correlate 10 100 1000 &
    c_pressure[1] c_pressure[2] c_pressure[3] &
    c_pressure[4] c_pressure[5] c_pressure[6] &
    type auto file stress_acf.dat ave running
```

### 3. Python分析自相关函数

```python
# autocorrelation_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def read_correlation(filename):
    """读取LAMMPS相关函数输出"""
    data = np.loadtxt(filename, comments='#')
    time = data[:, 1]  # 第2列是时间
    values = data[:, 2]  # 第3列是相关函数值
    return time, values

def compute_spectrum(correlation, dt):
    """计算功率谱 (FFT)"""
    # 补零到2的幂
    n = len(correlation)
    n_fft = 2**int(np.ceil(np.log2(n)))
    
    # FFT
    fft_vals = np.fft.fft(correlation, n=n_fft)
    freqs = np.fft.fftfreq(n_fft, dt)
    
    # 只取正频率
    pos_idx = freqs > 0
    return freqs[pos_idx], np.abs(fft_vals[pos_idx])**2

def green_kubo_viscosity(time, stress_acf, volume, temperature):
    """
    Green-Kubo公式计算粘度
    
    η = (V/kT) ∫_0^∞ <P_αβ(t)P_αβ(0)> dt
    """
    kB = 1.380649e-23  # J/K
    
    # 积分
    integral = integrate.trapz(stress_acf, time)
    
    # 转换为SI单位
    # stress_acf in (bar)² = (10^5 Pa)²
    # volume in Å³ = 10^-30 m³
    # time in fs = 10^-15 s
    integral_SI = integral * (1e5)**2 * 1e-30 * 1e-15  # Pa·s
    
    viscosity = volume * 1e-30 / (kB * temperature) * integral_SI
    
    return viscosity  # Pa·s

# 使用示例
time, vacf = read_correlation('vacf.dat')

# 计算振动态密度 (DOS)
freq, spectrum = compute_spectrum(vacf, dt=1e-15)  # dt in seconds

# 转换为THz
freq_THz = freq / 1e12

# 计算扩散系数 (Green-Kubo)
# D = (1/3) ∫_0^∞ <v(0)·v(t)> dt
D_integrand = vacf  # VACF已经是<v(0)·v(t)>
D = integrate.trapz(D_integrand, time * 1e-15) / 3  # m²/s
D_cm2s = D * 1e4  # cm²/s

print(f"Diffusion coefficient from VACF: {D_cm2s:.2e} cm²/s")

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(time, vacf, 'b-', linewidth=2)
axes[0].set_xlabel('Time (fs)')
axes[0].set_ylabel('VACF')
axes[0].set_title('Velocity Autocorrelation Function')
axes[0].grid(True)

axes[1].plot(freq_THz, spectrum, 'r-', linewidth=2)
axes[1].set_xlabel('Frequency (THz)')
axes[1].set_ylabel('Power Spectrum')
axes[1].set_title('Vibrational Density of States')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('vacf_analysis.png', dpi=300)
```

---

## 自由能计算

### 1. 伞形采样分析

```python
# umbrella_analysis.py
import numpy as np
import matplotlib.pyplot as plt

class WHAM:
    """加权直方图分析方法"""
    
    def __init__(self, k_B=0.001987):
        self.k_B = k_B  # kcal/mol/K
    
    def run_wham(self, histograms, bin_centers, k_springs, centers, T):
        """
        运行WHAM算法
        
        Parameters:
        -----------
        histograms : list of arrays
            每个窗口的直方图
        bin_centers : array
            bin中心位置
        k_springs : array
            每个窗口的弹簧常数
        centers : array
            每个窗口的平衡位置
        T : float
            温度 (K)
        """
        n_bins = len(bin_centers)
        n_windows = len(histograms)
        
        beta = 1.0 / (self.k_B * T)
        
        # 初始化自由能
        F = np.zeros(n_windows)
        
        # 迭代直到收敛
        for iteration in range(10000):
            F_old = F.copy()
            
            # 计算 unbiased 分布
            denom = np.zeros(n_bins)
            for i in range(n_windows):
                # 偏置势
                U_bias = 0.5 * k_springs[i] * (bin_centers - centers[i])**2
                denom += len(histograms[i]) * np.exp(-beta * (F[i] - U_bias))
            
            P_unbiased = np.sum(histograms, axis=0) / denom
            P_unbiased /= np.trapz(P_unbiased, bin_centers)
            
            # 更新自由能
            for i in range(n_windows):
                U_bias = 0.5 * k_springs[i] * (bin_centers - centers[i])**2
                F[i] = -np.log(np.trapz(P_unbiased * np.exp(-beta * U_bias), bin_centers)) / beta
            
            # 归一化
            F -= F[0]
            
            # 检查收敛
            if np.max(np.abs(F - F_old)) < 1e-6:
                print(f"WHAM converged after {iteration} iterations")
                break
        
        # 计算PMF
        pmf = -np.log(P_unbiased + 1e-10) / beta
        pmf -= pmf.min()
        
        return bin_centers, pmf

def analyze_umbrella_windows(window_files, centers, k_spring):
    """分析多个伞形采样窗口"""
    histograms = []
    all_data = []
    
    for i, filename in enumerate(window_files):
        data = np.loadtxt(filename)
        all_data.append(data)
        
        # 创建直方图
        hist, bin_edges = np.histogram(data, bins=100, range=(min(centers)-2, max(centers)+2))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        histograms.append(hist)
    
    return bin_centers, histograms, all_data

# 使用示例
window_files = [f'window_{i}.dat' for i in range(20)]
centers = np.linspace(2.0, 10.0, 20)
k_spring = 10.0

bin_centers, histograms, all_data = analyze_umbrella_windows(window_files, centers, k_spring)

wham = WHAM()
bin_centers, pmf = wham.run_wham(histograms, bin_centers, 
                                  [k_spring]*20, centers, T=300.0)

plt.plot(bin_centers, pmf, 'b-', linewidth=2)
plt.xlabel('Reaction Coordinate (Å)')
plt.ylabel('Free Energy (kcal/mol)')
plt.title('Potential of Mean Force')
plt.grid(True)
plt.savefig('pmf.png', dpi=300)
```

---

## 结构分析

### 1. 晶体结构分析

```python
# structure_analysis.py
import numpy as np
from ase import Atoms
from ase.io import read
from ase.geometry.analysis import Analysis
from ase.spacegroup import get_spacegroup

def analyze_crystal_structure(traj_file):
    """分析晶体结构演化"""
    traj = read(traj_file, index=':')
    
    results = {
        'lattice_params': [],
        'volumes': [],
        'spacegroups': []
    }
    
    for atoms in traj:
        # 晶格参数
        cell = atoms.get_cell()
        a, b, c = cell.lengths()
        alpha, beta, gamma = cell.angles()
        
        results['lattice_params'].append([a, b, c, alpha, beta, gamma])
        results['volumes'].append(atoms.get_volume())
        
        # 空间群 (可选)
        try:
            sg = get_spacegroup(atoms, symprec=1e-3)
            results['spacegroups'].append(sg.no)
        except:
            results['spacegroups'].append(None)
    
    return results

def compute_common_neighbor_analysis(atoms, cutoff=3.5):
    """
    共近邻分析 (CNA)
    
    Returns:
    --------
    dict: 不同CNA签名的计数
    """
    # CNA签名: (n_421, n_422, n_444)
    # FCC: (12, 0, 0) -> 12个(4,2,1)键
    # BCC: (6, 0, 8) -> 6个(4,2,1), 8个(4,4,4)
    # HCP: (6, 6, 0) -> 6个(4,2,1), 6个(4,2,2)
    
    n_atoms = len(atoms)
    cna_signatures = []
    
    for i in range(n_atoms):
        # 找到邻居
        neighbors = []
        for j in range(n_atoms):
            if i != j:
                r_ij = atoms.get_distance(i, j, mic=True)
                if r_ij < cutoff:
                    neighbors.append(j)
        
        # 计算共近邻
        n_421 = n_422 = n_444 = 0
        for j in neighbors:
            for k in neighbors:
                if j < k:
                    r_jk = atoms.get_distance(j, k, mic=True)
                    # 根据距离分类
                    if r_jk < cutoff:
                        n_common = len(set(neighbors) & 
                                      set([n for n in range(n_atoms) 
                                           if atoms.get_distance(j, n, mic=True) < cutoff]))
                        # 简化版本，实际需要更复杂计算
                        pass
        
        cna_signatures.append((n_421, n_422, n_444))
    
    return cna_signatures

def identify_local_structure(cna_signatures):
    """从CNA签名识别局部结构"""
    structures = []
    for sig in cna_signatures:
        n_421, n_422, n_444 = sig
        
        if n_421 == 12 and n_422 == 0 and n_444 == 0:
            structures.append('FCC')
        elif n_421 == 6 and n_422 == 6 and n_444 == 0:
            structures.append('HCP')
        elif n_421 == 6 and n_422 == 0 and n_444 == 8:
            structures.append('BCC')
        elif n_421 > 0:
            structures.append('Crystalline')
        else:
            structures.append('Other')
    
    return structures
```

### 2. 蛋白质结构分析

```python
# protein_analysis.py
import numpy as np
from ase.io import read

def compute_rmsd(ref_atoms, atoms):
    """计算RMSD"""
    pos_ref = ref_atoms.get_positions()
    pos = atoms.get_positions()
    
    # 对齐 (Kabsch算法简化版)
    pos_ref_centered = pos_ref - pos_ref.mean(axis=0)
    pos_centered = pos - pos.mean(axis=0)
    
    rmsd = np.sqrt(np.mean((pos_centered - pos_ref_centered)**2))
    return rmsd

def compute_radius_of_gyration(atoms):
    """计算回转半径"""
    positions = atoms.get_positions()
    masses = atoms.get_masses()
    
    com = np.average(positions, axis=0, weights=masses)
    
    Rg_squared = np.average(np.sum((positions - com)**2, axis=1), weights=masses)
    Rg = np.sqrt(Rg_squared)
    
    return Rg

def compute_end_to_end_distance(atoms):
    """计算端到端距离 (聚合物)"""
    positions = atoms.get_positions()
    return np.linalg.norm(positions[-1] - positions[0])

def compute_contact_map(atoms, cutoff=8.0):
    """计算接触图 (蛋白质)"""
    n_residues = len(set(atoms.get_residuenumbers()))
    contact_map = np.zeros((n_residues, n_residues))
    
    # 按残基分组
    residue_atoms = {}
    for i, res_id in enumerate(atoms.get_residuenumbers()):
        if res_id not in residue_atoms:
            residue_atoms[res_id] = []
        residue_atoms[res_id].append(i)
    
    # 计算残基间最小距离
    res_ids = sorted(residue_atoms.keys())
    for i, res_i in enumerate(res_ids):
        for j, res_j in enumerate(res_ids):
            if i != j:
                min_dist = np.inf
                for atom_i in residue_atoms[res_i]:
                    for atom_j in residue_atoms[res_j]:
                        dist = atoms.get_distance(atom_i, atom_j, mic=True)
                        min_dist = min(min_dist, dist)
                
                if min_dist < cutoff:
                    contact_map[i, j] = 1
    
    return contact_map
```

---

## 可视化工具

### 1. OVITO脚本

```python
# ovito_pipeline.py
from ovito.io import import_file, export_file
from ovito.modifiers import *

# 导入数据
pipeline = import_file("simulation.dump")

# 添加CNA修饰符
cna_modifier = CommonNeighborAnalysisModifier(cutoff=3.5)
pipeline.modifiers.append(cna_modifier)

# 添加颜色编码
color_modifier = ColorCodingModifier(
    property="Structure Type",
    gradient=ColorCodingModifier.Jet()
)
pipeline.modifiers.append(color_modifier)

# 导出
data = pipeline.compute(100)  # 第100帧
export_file(data, "output.png", "image")
```

### 2. VMD TCL脚本

```tcl
# vmd_analysis.tcl

# 加载轨迹
mol new system.psf
mol addfile trajectory.dcd waitfor all

# 设置表示
mol representation VDW 0.5 12.0
mol color Name
mol selection all
mol addrep top

# 计算RDF
set sel1 [atomselect top "type 1"]
set sel2 [atomselect top "type 2"]
measure gofr $sel1 $sel2 delta 0.1 rmax 10.0

# 渲染
render TachyonInternal frame_0000.tga
```

### 3. Matplotlib可视化

```python
# lammps_visualization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_trajectory(dump_file, frame=0):
    """可视化轨迹帧"""
    from ase.io import read
    
    traj = read(dump_file, index=':')
    atoms = traj[frame]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    positions = atoms.get_positions()
    types = atoms.get_atomic_numbers()
    
    # 按类型着色
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    
    for t in set(types):
        mask = types == t
        ax.scatter(positions[mask, 0], positions[mask, 1], positions[mask, 2],
                  c=colors[t % len(colors)], s=50, label=f'Type {t}')
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.legend()
    
    return fig
```

---

## Python后处理

### 1. 完整分析工作流

```python
# complete_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import pandas as pd

class LAMMPSAnalyzer:
    """LAMMPS模拟分析器"""
    
    def __init__(self, dump_file=None, log_file=None):
        self.dump_file = dump_file
        self.log_file = log_file
        self.trajectory = None
        self.thermo_data = None
        
        if dump_file:
            self.load_trajectory(dump_file)
        if log_file:
            self.load_log(log_file)
    
    def load_trajectory(self, filename):
        """加载轨迹"""
        self.trajectory = read(filename, index=':')
        print(f"Loaded {len(self.trajectory)} frames")
    
    def load_log(self, filename):
        """加载log文件"""
        # 解析log文件
        self.thermo_data = self._parse_log(filename)
    
    def _parse_log(self, filename):
        """解析LAMMPS log文件"""
        data = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        in_thermo = False
        header = []
        
        for line in lines:
            if line.startswith('Step'):
                header = line.split()
                in_thermo = True
            elif in_thermo and line.strip() and not line.startswith('Loop'):
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == len(header):
                        data.append(dict(zip(header, values)))
                except:
                    in_thermo = False
        
        return pd.DataFrame(data)
    
    def compute_rdf(self, frame=-1, bins=100, r_max=None):
        """计算RDF"""
        from ase.geometry.analysis import Analysis
        
        atoms = self.trajectory[frame]
        if r_max is None:
            r_max = min(atoms.get_cell().lengths()) / 2
        
        analyzer = Analysis([atoms])
        rdf = analyzer.get_rdf(rmax=r_max, nbins=bins)
        
        return rdf
    
    def compute_msd(self, atom_indices=None, dt=1.0):
        """计算MSD"""
        if atom_indices is None:
            atom_indices = range(len(self.trajectory[0]))
        
        n_frames = len(self.trajectory)
        msd = np.zeros(n_frames)
        
        ref_positions = self.trajectory[0].get_positions()[atom_indices]
        
        for i, atoms in enumerate(self.trajectory):
            positions = atoms.get_positions()[atom_indices]
            displacements = positions - ref_positions
            msd[i] = np.mean(np.sum(displacements**2, axis=1))
        
        time = np.arange(n_frames) * dt
        return time, msd
    
    def analyze_energetics(self):
        """分析能量"""
        if self.thermo_data is None:
            return None
        
        results = {
            'mean_pe': self.thermo_data['PotEng'].mean(),
            'mean_ke': self.thermo_data['TotEng'].mean(),
            'mean_temp': self.thermo_data['Temp'].mean(),
            'mean_press': self.thermo_data['Press'].mean()
        }
        
        return results
    
    def generate_report(self, output_dir='analysis_report'):
        """生成分析报告"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 能量分析
        if self.thermo_data is not None:
            self._plot_energetics(os.path.join(output_dir, 'energetics.png'))
        
        # RDF
        if self.trajectory is not None:
            self._plot_rdf(os.path.join(output_dir, 'rdf.png'))
        
        # MSD
        if len(self.trajectory) > 1:
            self._plot_msd(os.path.join(output_dir, 'msd.png'))
        
        print(f"Report saved to {output_dir}")
    
    def _plot_energetics(self, filename):
        """绘制能量图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0,0].plot(self.thermo_data['Step'], self.thermo_data['TotEng'])
        axes[0,0].set_ylabel('Total Energy')
        axes[0,0].set_title('Energy Evolution')
        
        axes[0,1].plot(self.thermo_data['Step'], self.thermo_data['Temp'])
        axes[0,1].set_ylabel('Temperature (K)')
        axes[0,1].set_title('Temperature')
        
        axes[1,0].plot(self.thermo_data['Step'], self.thermo_data['Press'])
        axes[1,0].set_ylabel('Pressure')
        axes[1,0].set_title('Pressure')
        
        axes[1,1].plot(self.thermo_data['Step'], self.thermo_data['Volume'])
        axes[1,1].set_ylabel('Volume')
        axes[1,1].set_title('Volume')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

# 使用示例
analyzer = LAMMPSAnalyzer(dump_file='simulation.dump', log_file='log.lammps')
analyzer.generate_report('my_analysis')
```

### 2. 批量分析工具

```python
# batch_analysis.py
import os
import glob
from concurrent.futures import ProcessPoolExecutor

def analyze_simulation(directory):
    """分析单个模拟目录"""
    dump_file = glob.glob(os.path.join(directory, '*.dump'))[0]
    log_file = glob.glob(os.path.join(directory, 'log.*'))[0]
    
    analyzer = LAMMPSAnalyzer(dump_file, log_file)
    
    # 生成报告
    output_dir = os.path.join(directory, 'analysis')
    analyzer.generate_report(output_dir)
    
    return directory, analyzer.analyze_energetics()

def batch_analyze(root_dir, n_workers=4):
    """批量分析多个模拟"""
    sim_dirs = [d for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d))]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(
            analyze_simulation, 
            [os.path.join(root_dir, d) for d in sim_dirs]
        ))
    
    return results
```

---

## 分析工具快速参考

| 分析类型 | LAMMPS内置 | Python后处理 | 可视化 |
|---------|-----------|-------------|--------|
| RDF | compute rdf + fix ave/time | 自定义脚本 | OVITO |
| MSD | compute msd + fix ave/time | ASE/Diffusion分析 | Matplotlib |
| VACF | compute vacf | SciPy积分 | Matplotlib |
| 密度分布 | compute chunk | NumPy直方图 | Matplotlib |
| 应力分析 | compute stress/atom | 张量分析 | OVITO |
| 自由能 | - | WHAM/MBAR | Matplotlib |
| 晶体结构 | compute cna | ASE/CNA | OVITO/VMD |
| 蛋白质结构 | - | MDAnalysis | VMD |

---

## 推荐阅读

- [LAMMPS Output Options](https://docs.lammps.org/Howto_output.html)
- [MDAnalysis User Guide](https://userguide.mdanalysis.org/)
- [ASE Documentation](https://wiki.fysik.dtu.dk/ase/)
- [OVITO User Manual](https://www.ovito.org/docs/current/)
