# DFT高级可视化脚本指南

本文档提供DFT计算结果的 publication-quality 可视化脚本，涵盖能带、态密度、电荷密度、声子谱等。

---

## 1. 能带结构可视化

### 1.1 基础能带图

```python
#!/usr/bin/env python3
# plot_bands.py - 标准能带结构绘制

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sys

def read_vasp_eigenval(filename='EIGENVAL'):
    """读取VASP EIGENVAL文件"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 读取基本信息
    nkpoints = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    
    kpoints = []
    bands = []
    
    line_idx = 7
    for _ in range(nkpoints):
        k = list(map(float, lines[line_idx].split()[:3]))
        kpoints.append(k)
        line_idx += 1
        
        band_energies = []
        for _ in range(nbands):
            energy = float(lines[line_idx].split()[1])
            band_energies.append(energy)
            line_idx += 1
        bands.append(band_energies)
        line_idx += 1
    
    return np.array(kpoints), np.array(bands)

def read_qe_bands(filename='bands.dat'):
    """读取QE能带数据"""
    data = np.loadtxt(filename)
    kpoints = data[:, 0]
    bands = data[:, 1:]
    return kpoints, bands

def calculate_kdistances(kpoints):
    """计算k点路径距离"""
    kdist = [0.0]
    for i in range(1, len(kpoints)):
        dk = np.linalg.norm(kpoints[i] - kpoints[i-1])
        kdist.append(kdist[-1] + dk)
    return np.array(kdist)

def plot_band_structure(kdist, bands, fermi_energy=0, 
                        klabels=None, kpositions=None,
                        energy_range=(-3, 3), output='bands.png'):
    """
    绘制 publication-quality 能带图
    
    Parameters:
    -----------
    kdist : array
        k点路径距离
    bands : array
        能带能量 (nkpoints, nbands)
    fermi_energy : float
        费米能级 (将设置为0)
    klabels : list
        高对称点标签
    kpositions : list
        高对称点位置
    energy_range : tuple
        能量显示范围
    output : str
        输出文件名
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 设置费米能级为0
    bands_shifted = bands - fermi_energy
    
    # 绘制能带
    for iband in range(bands.shape[1]):
        ax.plot(kdist, bands_shifted[:, iband], 
                color='#0066CC', linewidth=1.5, alpha=0.8)
    
    # 费米能级线
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # 高对称点标记
    if kpositions is not None:
        for pos in kpositions:
            ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    if klabels is not None and kpositions is not None:
        ax.set_xticks(kpositions)
        ax.set_xticklabels(klabels, fontsize=12)
    
    # 设置标签和范围
    ax.set_ylabel('Energy (eV)', fontsize=14)
    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_ylim(energy_range)
    
    # 美化
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"能带图已保存: {output}")
    plt.close()

# 主程序
if __name__ == '__main__':
    # 读取数据
    kpoints, bands = read_vasp_eigenval('EIGENVAL')
    
    # 计算k点距离
    kdist = calculate_kdistances(kpoints)
    
    # 定义高对称点 (示例: Si能带)
    klabels = ['L', 'Γ', 'X', 'U', 'Γ']
    # 根据实际k点路径确定位置
    kpositions = [0, kdist[25], kdist[50], kdist[75], kdist[-1]]
    
    # 从OUTCAR读取费米能级
    fermi_energy = 6.245  # 请替换为实际值
    
    # 绘制
    plot_band_structure(kdist, bands, fermi_energy, 
                       klabels, kpositions,
                       energy_range=(-15, 10),
                       output='band_structure.png')
```

### 1.2 投影能带图 (轨道/原子投影)

```python
#!/usr/bin/env python3
# plot_fatbands.py - 投影能带图 (Fatbands)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

def plot_fatbands(kdist, bands, projections, fermi_energy=0,
                  energy_range=(-3, 3), cmap='RdYlBu',
                  output='fatbands.png'):
    """
    绘制投影能带图
    
    projections: array (nkpoints, nbands) - 投影权重
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    bands_shifted = bands - fermi_energy
    
    # 创建线段集合
    for iband in range(bands.shape[1]):
        # 创建线段
        points = np.array([kdist, bands_shifted[:, iband]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 根据投影权重设置颜色
        weights = projections[:, iband]
        
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(weights)
        lc.set_linewidth(3)
        ax.add_collection(lc)
    
    # 颜色条
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('Projection Weight', fontsize=12)
    
    # 费米能级
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlabel('k-path', fontsize=14)
    ax.set_ylabel('Energy (eV)', fontsize=14)
    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_ylim(energy_range)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()

# 示例: 读取PROCAR并绘制投影
from procar import Procar  # 需要安装pyprocar或使用自定义解析

# proc = Procar.procarFile('PROCAR', 'PROCAR')
# projections = proc.get_projections(atom=0, orbital='pz')
```

### 1.3 自旋极化能带

```python
def plot_spin_polarized_bands(kdist, bands_up, bands_dn, fermi_energy=0,
                              energy_range=(-3, 3), output='bands_spin.png'):
    """
    绘制自旋极化能带
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    bands_up_shifted = bands_up - fermi_energy
    bands_dn_shifted = bands_dn - fermi_energy
    
    # 自旋向上 (蓝色)
    for iband in range(bands_up.shape[1]):
        ax.plot(kdist, bands_up_shifted[:, iband], 
                color='blue', linewidth=1.5, alpha=0.7)
    
    # 自旋向下 (红色)
    for iband in range(bands_dn.shape[1]):
        ax.plot(kdist, bands_dn_shifted[:, iband], 
                color='red', linewidth=1.5, alpha=0.7)
    
    # 费米能级
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # 图例
    ax.plot([], [], color='blue', linewidth=2, label='Spin ↑')
    ax.plot([], [], color='red', linewidth=2, label='Spin ↓')
    ax.legend(loc='upper right', fontsize=12)
    
    ax.set_ylabel('Energy (eV)', fontsize=14)
    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_ylim(energy_range)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 2. 态密度可视化

### 2.1 总态密度与投影态密度

```python
#!/usr/bin/env python3
# plot_dos.py - 态密度可视化

import numpy as np
import matplotlib.pyplot as plt

def read_vasp_dos(filename='DOSCAR'):
    """读取VASP DOSCAR"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 读取能量网格数
    nedos = int(lines[5].split()[2])
    fermi_energy = float(lines[5].split()[3])
    
    # 读取TDOS
    energy = []
    dos_total = []
    dos_integrated = []
    
    for i in range(6, 6 + nedos):
        values = list(map(float, lines[i].split()))
        energy.append(values[0])
        dos_total.append(values[1])
        dos_integrated.append(values[2])
    
    return np.array(energy), np.array(dos_total), fermi_energy

def read_pdos_files(filenames):
    """读取多个PDOS文件"""
    pdos_data = {}
    for filename in filenames:
        data = np.loadtxt(filename)
        energy = data[:, 0]
        dos = data[:, 1:]
        label = filename.split('.')[0]
        pdos_data[label] = (energy, dos)
    return pdos_data

def plot_dos(energy, dos_total, fermi_energy=0, 
             pdos_data=None, energy_range=(-10, 5),
             output='dos.png'):
    """
    绘制态密度图
    
    Parameters:
    -----------
    energy : array
        能量网格
    dos_total : array
        总态密度
    fermi_energy : float
        费米能级
    pdos_data : dict
        投影态密度数据 {label: (energy, dos)}
    energy_range : tuple
        能量显示范围
    output : str
        输出文件名
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # 设置费米能级为0
    energy_shifted = energy - fermi_energy
    mask = (energy_shifted >= energy_range[0]) & (energy_shifted <= energy_range[1])
    
    # 绘制总DOS
    ax.fill_between(energy_shifted[mask], 0, dos_total[mask], 
                    alpha=0.3, color='gray', label='Total DOS')
    ax.plot(energy_shifted[mask], dos_total[mask], 
            color='black', linewidth=1.5)
    
    # 绘制投影DOS
    if pdos_data is not None:
        colors = plt.cm.Set2(np.linspace(0, 1, len(pdos_data)))
        for (label, (e, dos)), color in zip(pdos_data.items(), colors):
            e_shifted = e - fermi_energy
            # 通常PDOS有多列 (s, p, d等)
            if dos.ndim > 1:
                dos_sum = np.sum(dos, axis=1)
            else:
                dos_sum = dos
            ax.plot(e_shifted, dos_sum, label=label, 
                   color=color, linewidth=2)
    
    # 费米能级
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(0.2, ax.get_ylim()[1]*0.9, '$E_F$', fontsize=12, color='red')
    
    ax.set_xlabel('Energy (eV)', fontsize=14)
    ax.set_ylabel('DOS (states/eV)', fontsize=14)
    ax.set_xlim(energy_range)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"DOS图已保存: {output}")
    plt.close()

# 主程序
if __name__ == '__main__':
    energy, dos_total, ef = read_vasp_dos('DOSCAR')
    
    # 读取PDOS (可选)
    # pdos_files = ['PDOS_Si_s.dat', 'PDOS_Si_p.dat']
    # pdos_data = read_pdos_files(pdos_files)
    
    plot_dos(energy, dos_total, ef, 
             # pdos_data=pdos_data,
             energy_range=(-15, 10),
             output='dos.png')
```

### 2.2 分波态密度 (轨道分解)

```python
def plot_orbital_dos(energy, pdos_dict, fermi_energy=0,
                     energy_range=(-10, 5), output='pdos_orbital.png'):
    """
    绘制轨道分解的PDOS
    
    pdos_dict: {orbital: dos_array}
        如: {'s': dos_s, 'p': dos_p, 'd': dos_d}
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    energy_shifted = energy - fermi_energy
    mask = (energy_shifted >= energy_range[0]) & (energy_shifted <= energy_range[1])
    
    colors = {'s': '#1f77b4', 'p': '#ff7f0e', 'd': '#2ca02c', 'f': '#d62728'}
    
    for orbital, dos in pdos_dict.items():
        color = colors.get(orbital, 'gray')
        ax.fill_between(energy_shifted[mask], 0, dos[mask],
                       alpha=0.4, color=color)
        ax.plot(energy_shifted[mask], dos[mask], 
               label=f'{orbital} orbital', color=color, linewidth=2)
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Energy (eV)', fontsize=14)
    ax.set_ylabel('PDOS (states/eV)', fontsize=14)
    ax.set_xlim(energy_range)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 3. 电荷密度可视化

### 3.1 2D电荷密度切片

```python
#!/usr/bin/env python3
# plot_charge_density.py - 电荷密度可视化

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def read_vasp_chgcar(filename='CHGCAR'):
    """读取VASP CHGCAR文件"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 读取晶格矢量
    scale = float(lines[1])
    lattice = np.array([list(map(float, lines[i].split())) 
                        for i in range(2, 5)]) * scale
    
    # 读取网格尺寸
    ngxf, ngyf, ngzf = map(int, lines[5].split())
    
    # 读取电荷密度
    chg_data = []
    for line in lines[6:]:
        chg_data.extend(map(float, line.split()))
        if len(chg_data) >= ngxf * ngyf * ngzf:
            break
    
    chg = np.array(chg_data[:ngxf*ngyf*ngzf]).reshape((ngxf, ngyf, ngzf))
    
    return lattice, chg

def plot_charge_slice(chg, lattice, plane='xy', position=0.5,
                      cmap='RdYlBu_r', output='charge_slice.png'):
    """
    绘制电荷密度切片
    
    Parameters:
    -----------
    chg : 3D array
        电荷密度数据
    lattice : 3x3 array
        晶格矢量
    plane : str
        'xy', 'yz', 或 'xz'
    position : float
        切片位置 (0-1分数坐标)
    cmap : str
        颜色映射
    output : str
        输出文件名
    """
    nx, ny, nz = chg.shape
    
    if plane == 'xy':
        iz = int(position * nz)
        slice_data = chg[:, :, iz].T
        xlabel, ylabel = 'x (Å)', 'y (Å)'
        extent = [0, lattice[0,0], 0, lattice[1,1]]
    elif plane == 'yz':
        ix = int(position * nx)
        slice_data = chg[ix, :, :].T
        xlabel, ylabel = 'y (Å)', 'z (Å)'
        extent = [0, lattice[1,1], 0, lattice[2,2]]
    elif plane == 'xz':
        iy = int(position * ny)
        slice_data = chg[:, iy, :].T
        xlabel, ylabel = 'x (Å)', 'z (Å)'
        extent = [0, lattice[0,0], 0, lattice[2,2]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(slice_data, origin='lower', cmap=cmap,
                   extent=extent, aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Charge Density (e/Å³)', fontsize=12)
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'Charge Density Slice ({plane} plane)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"电荷密度图已保存: {output}")
    plt.close()

def plot_charge_difference(chg1, chg2, lattice, plane='xy', position=0.5,
                          output='charge_diff.png'):
    """
    绘制差分电荷密度
    """
    diff = chg1 - chg2
    
    nx, ny, nz = diff.shape
    if plane == 'xy':
        iz = int(position * nz)
        slice_data = diff[:, :, iz].T
        extent = [0, lattice[0,0], 0, lattice[1,1]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmax = np.abs(slice_data).max()
    im = ax.imshow(slice_data, origin='lower', cmap='RdBu_r',
                   extent=extent, aspect='equal',
                   vmin=-vmax, vmax=vmax)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Δρ (e/Å³)', fontsize=12)
    
    ax.set_xlabel('x (Å)', fontsize=14)
    ax.set_ylabel('y (Å)', fontsize=14)
    ax.set_title('Charge Density Difference', fontsize=14)
    
    # 添加等值线
    ax.contour(slice_data, levels=10, colors='black', alpha=0.3,
               extent=extent, linewidths=0.5)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()

# 主程序
if __name__ == '__main__':
    lattice, chg = read_vasp_chgcar('CHGCAR')
    
    # 绘制xy平面切片 (z=0.5)
    plot_charge_slice(chg, lattice, plane='xy', position=0.5,
                     output='charge_xy.png')
```

### 3.2 3D等值面 (需要mayavi或plotly)

```python
def plot_charge_isosurface(chg, lattice, isovalue=0.05):
    """
    使用mayavi绘制3D等值面
    (需要安装mayavi: pip install mayavi)
    """
    try:
        from mayavi import mlab
    except ImportError:
        print("请安装mayavi: pip install mayavi")
        return
    
    nx, ny, nz = chg.shape
    
    # 创建网格
    x = np.linspace(0, lattice[0,0], nx)
    y = np.linspace(0, lattice[1,1], ny)
    z = np.linspace(0, lattice[2,2], nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 绘制等值面
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.contour3d(X, Y, Z, chg, contours=[isovalue], 
                   color=(0.2, 0.4, 0.8), opacity=0.5)
    
    mlab.axes(xlabel='x', ylabel='y', zlabel='z')
    mlab.colorbar(title='ρ (e/Å³)')
    mlab.show()
```

---

## 4. 声子谱可视化

### 4.1 声子色散曲线

```python
#!/usr/bin/env python3
# plot_phonon.py - 声子谱可视化

def plot_phonon_bands(qpath, frequencies, qpositions=None, qlabels=None,
                      energy_range=None, output='phonon_bands.png'):
    """
    绘制声子色散曲线
    
    Parameters:
    -----------
    qpath : array
        q点路径坐标
    frequencies : array (nqpoints, nbranches)
        声子频率 (THz)
    qpositions : list
        高对称点位置
    qlabels : list
        高对称点标签
    energy_range : tuple
        频率范围 (THz)
    output : str
        输出文件名
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    nbranches = frequencies.shape[1]
    
    # 绘制声子支
    for ibranch in range(nbranches):
        ax.plot(qpath, frequencies[:, ibranch], 
                color='blue', linewidth=1.5, alpha=0.7)
    
    # 零线
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    # 高对称点标记
    if qpositions is not None:
        for pos in qpositions:
            ax.axvline(x=pos, color='gray', linestyle='-', 
                      linewidth=0.8, alpha=0.5)
    
    if qlabels is not None and qpositions is not None:
        ax.set_xticks(qpositions)
        ax.set_xticklabels(qlabels, fontsize=12)
    
    ax.set_xlabel('q-path', fontsize=14)
    ax.set_ylabel('Frequency (THz)', fontsize=14)
    ax.set_xlim(qpath[0], qpath[-1])
    
    if energy_range:
        ax.set_ylim(energy_range)
    else:
        ax.set_ylim(0, frequencies.max() * 1.1)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"声子谱已保存: {output}")
    plt.close()

def plot_phonon_dos(freq_dos, dos, output='phonon_dos.png'):
    """
    绘制声子态密度
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.fill_between(freq_dos, 0, dos, alpha=0.5, color='blue')
    ax.plot(freq_dos, dos, color='darkblue', linewidth=2)
    
    ax.set_xlabel('Frequency (THz)', fontsize=14)
    ax.set_ylabel('Phonon DOS', fontsize=14)
    ax.set_xlim(0, freq_dos.max())
    ax.set_ylim(0, dos.max() * 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()

# 组合图 (色散+态密度)
def plot_phonon_combined(qpath, frequencies, freq_dos, dos,
                        output='phonon_combined.png'):
    """
    声子色散+态密度组合图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                   gridspec_kw={'width_ratios': [3, 1]})
    
    # 色散图
    for ibranch in range(frequencies.shape[1]):
        ax1.plot(qpath, frequencies[:, ibranch], 
                color='blue', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('q-path', fontsize=14)
    ax1.set_ylabel('Frequency (THz)', fontsize=14)
    ax1.set_xlim(qpath[0], qpath[-1])
    ax1.set_ylim(0, frequencies.max() * 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # DOS图
    ax2.fill_betweenx(freq_dos, 0, dos, alpha=0.5, color='blue')
    ax2.plot(dos, freq_dos, color='darkblue', linewidth=2)
    ax2.set_ylim(0, frequencies.max() * 1.1)
    ax2.set_xlabel('DOS', fontsize=14)
    ax2.set_yticklabels([])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 5. 热图与等高线图

### 5.1 能带热图

```python
def plot_band_heatmap(kdist, bands, fermi_energy=0,
                      energy_range=(-3, 3), output='band_heatmap.png'):
    """
    能带热图 (适用于大量能带)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bands_shifted = bands - fermi_energy
    
    # 创建2D直方图
    extent = [kdist[0], kdist[-1], energy_range[0], energy_range[1]]
    
    # 将所有能带点合并
    all_k = np.repeat(kdist, bands.shape[1])
    all_e = bands_shifted.flatten()
    
    # 2D直方图
    h, xedges, yedges = np.histogram2d(all_k, all_e, bins=(200, 200))
    
    im = ax.imshow(h.T, origin='lower', cmap='hot',
                   extent=extent, aspect='auto')
    
    plt.colorbar(im, ax=ax, label='Intensity')
    ax.set_xlabel('k-path')
    ax.set_ylabel('Energy (eV)')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
```

---

## 6. 使用说明

### 6.1 安装依赖

```bash
# 基础依赖
pip install numpy matplotlib scipy

# 3D可视化 (可选)
pip install mayavi plotly

# ASE支持 (读取结构)
pip install ase
```

### 6.2 脚本调用示例

```bash
# 能带图
python plot_bands.py --eigenval EIGENVAL --fermi 6.245 --output bands.png

# DOS图
python plot_dos.py --doscar DOSCAR --range -15 10 --output dos.png

# 声子谱
python plot_phonon.py --freq si.freq.gp --output phonon.png

# 电荷密度
python plot_charge_density.py --chgcar CHGCAR --plane xy --output charge.png
```

### 6.3 批量处理

```bash
#!/bin/bash
# batch_plot.sh - 批量绘图

for dir in */; do
    cd $dir
    
    if [ -f EIGENVAL ]; then
        python ../plot_bands.py
    fi
    
    if [ -f DOSCAR ]; then
        python ../plot_dos.py
    fi
    
    cd ..
done
```

---

*最后更新: 2026-03-08*
