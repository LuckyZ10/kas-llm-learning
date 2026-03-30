# 表面态分析

## 概述

固体表面打破了体材料的周期性，导致电子态在表面附近局域化。这些表面态在催化、电子输运、拓扑绝缘体等研究中至关重要。本指南涵盖表面态的理论基础、识别方法和计算流程。

---

## 理论基础

### 表面态分类

#### 1. Tamm 态
- 起源于表面势的突然中断
- 存在于表面原子层
- 对表面结构敏感

#### 2. Shockley 态
- 起源于能带交叉的表面投影
- 存在于近表面的势阱中
- 对表面势形状敏感

#### 3. 悬挂键态 (Dangling Bond States)
- 半导体表面特征
- 未饱和的表面键形成
- 常见于Si(111) 2×1, GaAs(110)等

#### 4. 拓扑表面态
- 受时间反演对称性保护
- 狄拉克锥色散
- 自旋-动量锁定

### 表面态特征

**能量位置**:
```
E_surface ∈ 带隙 (通常)
E_bulk ∈ 能带 (连续)
```

**空间分布**:
```
|ψ_surface(z)|² ∝ exp(-z/ξ)   # 指数衰减进入体相
|ψ_bulk(z)|² = 常数            # 扩展态
```

其中 **ξ** 是表面态穿透深度，通常为 1-3 原子层。

---

## 计算方法

### 1. 层投影能带结构 (Layer Projected Bands)

#### VASP 设置

```bash
# INCAR
LORBIT = 11           # 输出投影态密度
NEDOS = 2000          # 精细能量网格

# 特殊设置用于层投影
LPARD = .TRUE.        # 波函数投影
NBMOD = -2            # 投影到原子/层
EINT = -5 5           # 能量范围
```

#### 后处理脚本

```python
#!/usr/bin/env python3
"""
layer_projected_bands.py - 层投影能带分析
"""
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Vasprun, Procar
from pymatgen.electronic_structure.core import Spin

def calculate_layer_weights(procar, structure, n_layers=5):
    """
    计算各原子层对能带的贡献权重
    
    Args:
        procar: PROCAR数据
        structure: 结构对象
        n_layers: 沿z方向分层数
    """
    # 按z坐标分层
    z_coords = [site.frac_coords[2] for site in structure]
    z_min, z_max = min(z_coords), max(z_coords)
    layer_boundaries = np.linspace(z_min, z_max, n_layers + 1)
    
    # 分配原子到层
    atom_to_layer = []
    for z in z_coords:
        for i in range(n_layers):
            if layer_boundaries[i] <= z <= layer_boundaries[i+1]:
                atom_to_layer.append(i)
                break
    
    # 计算层投影权重
    layer_weights = np.zeros((n_layers, procar.data.shape[1], procar.data.shape[2]))
    
    for atom_idx, layer_idx in enumerate(atom_to_layer):
        # 累加该原子对所有k点、能带的贡献
        layer_weights[layer_idx] += procar.data[atom_idx]
    
    return layer_weights, atom_to_layer

def identify_surface_states(kpoints, energies, layer_weights, 
                           surface_layer=0, threshold=0.5):
    """
    识别表面态
    
    Args:
        threshold: 表面层权重阈值
    """
    surface_states = []
    
    for ik, k in enumerate(kpoints):
        for ib, E in enumerate(energies[ik]):
            # 检查表面层权重
            surface_weight = layer_weights[surface_layer, ik, ib]
            total_weight = np.sum(layer_weights[:, ik, ib])
            
            if total_weight > 0:
                ratio = surface_weight / total_weight
                if ratio > threshold:
                    surface_states.append({
                        'kpoint': k,
                        'energy': E,
                        'surface_ratio': ratio,
                        'k_idx': ik,
                        'band_idx': ib
                    })
    
    return surface_states

# 可视化
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_layer_projected_bands(kpath, energies, layer_weights, 
                               layer_idx=0, cmap='Reds', output='layer_bands.png'):
    """
    绘制层投影能带图
    
    颜色表示该层对态的贡献权重
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 创建颜色映射
    weights = layer_weights[layer_idx]
    norm = plt.Normalize(vmin=0, vmax=1)
    
    for ib in range(energies.shape[1]):
        sc = ax.scatter(kpath, energies[:, ib], 
                       c=weights[:, ib],
                       cmap=cmap,
                       s=10,
                       norm=norm,
                       alpha=0.8)
    
    ax.set_xlabel('k-path')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(f'Layer {layer_idx} Projected Band Structure')
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Layer Weight')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

if __name__ == '__main__':
    # 使用示例
    from pymatgen.io.vasp import Vasprun
    
    vrun = Vasprun('vasprun.xml')
    procar = Procar('PROCAR')
    
    layer_weights, atom_map = calculate_layer_weights(
        procar, vrun.final_structure, n_layers=7
    )
    
    # 绘制表面层(顶层)投影
    kpath = np.linspace(0, 1, len(vrun.actual_kpoints))
    plot_layer_projected_bands(kpath, vrun.eigenvalues[Spin.up][:, :, 0],
                               layer_weights, layer_idx=0)
```

### 2. 波函数分析 (Charge Density Localization)

```python
#!/usr/bin/env python3
"""
surface_state_wfn.py - 表面态波函数分析
"""
import numpy as np
from ase.io import read, write
from ase.io.cube import read_cube_data
from scipy.ndimage import gaussian_filter1d

def analyze_state_localization(cube_file, axis=2, surface_threshold=0.8):
    """
    分析波函数沿特定方向的局域化程度
    
    Args:
        cube_file: 波函数密度cube文件
        axis: 分析方向 (0=x, 1=y, 2=z)
        surface_threshold: 表面权重阈值
    """
    # 读取cube数据
    data, atoms = read_cube_data(cube_file)
    
    # 沿指定轴积分
    axes_to_sum = tuple(i for i in range(3) if i != axis)
    profile = np.sum(data, axis=axes_to_sum)
    
    # 归一化
    profile = profile / np.sum(profile)
    
    # 计算质心位置
    grid_points = np.arange(len(profile))
    centroid = np.sum(grid_points * profile)
    
    # 计算展宽 (标准差)
    variance = np.sum((grid_points - centroid)**2 * profile)
    spread = np.sqrt(variance)
    
    # 判断是否为表面态
    n_points = len(profile)
    surface_region = 0.15  # 边缘15%认为是表面
    
    is_surface = (centroid < n_points * surface_region or 
                  centroid > n_points * (1 - surface_region))
    
    # 计算表面权重
    if centroid < n_points * 0.5:
        surface_weight = np.sum(profile[:int(n_points * surface_region)])
    else:
        surface_weight = np.sum(profile[int(n_points * (1 - surface_region)):])
    
    return {
        'centroid': centroid,
        'spread': spread,
        'is_surface': is_surface and surface_weight > surface_threshold,
        'surface_weight': surface_weight,
        'profile': profile
    }

def identify_surface_states_from_wfn(cube_files, axis=2):
    """
    从多个波函数文件中识别表面态
    
    Returns:
        list: 表面态文件列表及特征
    """
    surface_states = []
    
    for cube_file in cube_files:
        result = analyze_state_localization(cube_file, axis)
        
        if result['is_surface']:
            surface_states.append({
                'file': cube_file,
                'surface_weight': result['surface_weight'],
                'spread': result['spread']
            })
    
    return surface_states

# 示例使用
if __name__ == '__main__':
    import glob
    
    cube_files = glob.glob('WAVECAR_*.cube')
    surface_states = identify_surface_states_from_wfn(cube_files)
    
    print("识别到的表面态:")
    for state in surface_states:
        print(f"  {state['file']}: 表面权重={state['surface_weight']:.2%}, "
              f"展宽={state['spread']:.2f} grid")
```

### 3. 能带反折叠 (Band Unfolding)

表面超胞计算需要将能带反折叠到原胞布里渊区：

```python
#!/usr/bin/env python3
"""
band_unfolding.py - 能带反折叠用于表面态分析
"""
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.bandstructure import HighSymmKpath

class BandUnfolder:
    """能带反折叠类"""
    
    def __init__(self, primitive_structure, supercell_structure, transformation_matrix):
        """
        Args:
            primitive_structure: 原胞结构
            supercell_structure: 超胞结构
            transformation_matrix: 3×3超胞变换矩阵
        """
        self.prim = primitive_structure
        self.sc = supercell_structure
        self.M = np.array(transformation_matrix)
        
    def calculate_spectral_weight(self, k_prim, k_sc, eigenvectors):
        """
        计算谱权重
        
        P(K,k) = |⟨K,k|K⟩|²
        """
        # 计算折叠权重
        # 这是简化版本，实际需要处理平面波展开
        
        # 检查k点是否折叠到原胞
        k_diff = k_sc - np.dot(self.M.T, k_prim)
        k_diff -= np.round(k_diff)  # 到第一布里渊区
        
        is_folded = np.allclose(k_diff, 0, atol=1e-4)
        
        if is_folded:
            # 简化: 返回均匀权重
            return 1.0 / np.linalg.det(self.M)
        else:
            return 0.0
    
    def unfold_bands(self, kpoints_sc, energies_sc, eigenvectors_sc,
                     kpoints_prim_grid):
        """
        执行能带反折叠
        
        Returns:
            (kpoints_prim, energies_unfolded, spectral_weights)
        """
        unfolded_data = []
        
        for ik_sc, k_sc in enumerate(kpoints_sc):
            for ib, E in enumerate(energies_sc[ik_sc]):
                # 找到对应的原胞k点
                for k_prim in kpoints_prim_grid:
                    weight = self.calculate_spectral_weight(
                        k_prim, k_sc, eigenvectors_sc[ik_sc][ib]
                    )
                    
                    if weight > 1e-6:
                        unfolded_data.append({
                            'k_prim': k_prim,
                            'energy': E,
                            'weight': weight
                        })
        
        return unfolded_data

# 使用BandUP软件接口
def run_bandup_interface():
    """
    使用BandUP软件进行专业反折叠
    
    需要预先安装BandUP:
    https://github.com/band-unfolding/bandup
    """
    instructions = """
    BandUP使用步骤:
    
    1. 准备文件:
       - scf.in (原胞自洽)
       - bands.in (原胞能带)
       - scf_supercell.in (超胞自洽)
       - bands_supercell.in (超胞能带)
    
    2. 运行BandUP:
       bandup unfolded_band_structure
       
    3. 输出:
       - unfolded_EBS.dat: 反折叠能带数据
    """
    print(instructions)
```

### 4. 费米面分析 (2D 切片)

```python
#!/usr/bin/env python3
"""
fermi_surface_analysis.py - 表面态费米面分析
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_surface_states_fermi_surface(fermi_data, energy_window=0.1):
    """
    提取费米面附近的表面态
    
    Args:
        fermi_data: (kx, ky, E, weight) 数组
        energy_window: 能量窗口 (eV)
    """
    kx, ky, E, weight = fermi_data
    
    # 筛选费米面附近
    mask = np.abs(E) < energy_window
    
    # 筛选高权重 (表面态特征)
    weight_threshold = 0.3
    surface_mask = mask & (weight > weight_threshold)
    
    return {
        'kx': kx[surface_mask],
        'ky': ky[surface_mask],
        'weight': weight[surface_mask],
        'E': E[surface_mask]
    }

def plot_2d_surface_states(kx, ky, weight, title='Surface State Fermi Surface'):
    """绘制2D表面态费米面"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    scatter = ax.scatter(kx, ky, c=weight, cmap='hot', 
                        s=20, alpha=0.6)
    
    ax.set_xlabel('$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('$k_y$ (Å$^{-1}$)')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    plt.colorbar(scatter, label='Surface Weight')
    plt.tight_layout()
    plt.savefig('fermi_surface_2d.png', dpi=300)
    plt.close()
```

---

## 拓扑表面态

### 特征识别

```python
#!/usr/bin/env python3
"""
topological_surface_states.py - 拓扑表面态分析
"""
import numpy as np

def analyze_dirac_cone(kpoints, energies, k_tol=0.05, E_tol=0.1):
    """
    识别狄拉克锥特征
    
    Args:
        k_tol: k点容差 (Å⁻¹)
        E_tol: 能量容差 (eV)
    """
    # 寻找能带交叉点
    crossings = []
    
    for ik in range(len(kpoints) - 1):
        for ib1 in range(len(energies[ik])):
            for ib2 in range(ib1 + 1, len(energies[ik])):
                E1, E2 = energies[ik][ib1], energies[ik][ib2]
                
                # 检查简并
                if abs(E1 - E2) < E_tol:
                    crossings.append({
                        'k': kpoints[ik],
                        'E': (E1 + E2) / 2,
                        'bands': (ib1, ib2)
                    })
    
    # 验证线性色散
    dirac_points = []
    for cross in crossings:
        k_idx = np.argmin(np.linalg.norm(kpoints - cross['k'], axis=1))
        
        # 计算附近斜率
        if 0 < k_idx < len(kpoints) - 1:
            slope_before = (energies[k_idx] - energies[k_idx-1]) / \
                          (kpoints[k_idx] - kpoints[k_idx-1])
            slope_after = (energies[k_idx+1] - energies[k_idx]) / \
                         (kpoints[k_idx+1] - kpoints[k_idx])
            
            # 检查是否符号相反 (X型交叉)
            for ib1, ib2 in [(cross['bands'][0], cross['bands'][1])]:
                if slope_before[ib1] * slope_after[ib1] < 0:
                    dirac_points.append(cross)
    
    return dirac_points

def calculate_spin_texture(kpoints, energies, spin_data):
    """
    计算自旋纹理 (用于验证拓扑保护)
    
    Returns:
        自旋-动量锁定特征
    """
    spin_texture = []
    
    for ik, k in enumerate(kpoints):
        for ib, E in enumerate(energies[ik]):
            if abs(E) < 0.5:  # 费米面附近
                Sx, Sy, Sz = spin_data[ik][ib]
                
                # 计算自旋-动量夹角
                k_vec = np.array(k[:2])  # 2D投影
                S_vec = np.array([Sx, Sy])
                
                if np.linalg.norm(k_vec) > 0 and np.linalg.norm(S_vec) > 0:
                    cos_theta = np.dot(k_vec, S_vec) / \
                               (np.linalg.norm(k_vec) * np.linalg.norm(S_vec))
                    
                    spin_texture.append({
                        'k': k,
                        'spin': [Sx, Sy, Sz],
                        'angle': np.arccos(np.clip(cos_theta, -1, 1))
                    })
    
    return spin_texture
```

---

## 实际案例

### 案例1: Si(111) 表面态

```bash
# VASP计算设置
# Si(111) 2×1重构表面

# INCAR
ISMEAR = 0
SIGMA = 0.05
ISYM = 0            # 关闭对称性
EDIFF = 1E-7
ENCUT = 500

# 表面态分析
LORBIT = 11         # 投影态密度
LVHAR = .TRUE.      # 输出静电势
```

### 案例2: 拓扑绝缘体 Bi₂Se₃

```bash
# 关键设置 (强SOC)
LSORBIT = .TRUE.
SAXIS = 0 0 1       # z方向自旋

# 能带计算
ICHARG = 11
LORBIT = 11
```

---

## 参考资源

- [VASP表面计算Wiki](https://www.vasp.at/wiki/index.php/Surface_calculations)
- [BandUP反折叠工具](https://github.com/band-unfolding/bandup)
- [Wannier90表面态](http://www.wannier.org/)
