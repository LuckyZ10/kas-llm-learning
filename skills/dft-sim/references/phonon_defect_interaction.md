# 声子-缺陷相互作用

## 概述

晶体中的点缺陷（空位、间隙、替位原子）会打破晶格的周期性，导致局域化的声子模式。这些局域声子不仅影响缺陷的热力学稳定性，还在电子-声子相互作用、载流子散射和非辐射复合等过程中起关键作用。

---

## 理论基础

### 缺陷引起的晶格畸变

点缺陷周围的晶格弛豫可用**畸变势**描述：

```
ΔE_distortion = Σ_i ½·k_i·(Δr_i)²
```

其中：
- **Δr_i**: 第i个原子相对于完美晶格的位移
- **k_i**: 有效力常数

### 局域声子模式

缺陷引入三类声子模式：

1. **共振模式 (Resonance modes)**: 频率落入体材料声子带内
2. **隙模式 (Gap modes)**: 频率落入声子带隙
3. **局域模式 (Localized modes)**: 高频（轻杂质）或低频（重杂质），位于带外

### 振动熵贡献

缺陷形成能的温度依赖：

```
E_f(T) = E_f(0K) - T·ΔS_vib

ΔS_vib = -k_B · Σ_λ ln(2·sinh(ℏω_λ/2k_BT))
```

**重要性**: 振动熵贡献可达 0.1-0.5 eV @ 1000K

---

## 计算方法

### 1. 超胞有限位移法

#### 原理

在缺陷超胞中原子位移，计算力常数矩阵：

```
C_ij = ∂²E/∂u_i∂u_j ≈ [F_i(+δu_j) - F_i(-δu_j)] / (2·δu_j)
```

#### VASP 设置

```bash
# INCAR.phonon_defect
IBRION = 6            # 有限位移法
NFREE = 2             # 双向位移
POTIM = 0.015         # 位移幅度 (Å)
NSW = 1               # 离子步数 (自动计算)
EDIFF = 1E-8          # 高精度电子收敛

# 关键: 仅缺陷近邻原子参与计算
# 使用 ICONST 文件指定活性原子
```

#### ICONST 文件 (选择性位移)

```
# 仅允许缺陷第一近邻原子移动
# 格式: LR 原子索引范围
# 例如: 仅原子1-12 (缺陷周围)
LR
1 12
```

### 2. 冻声子法 (Frozen Phonon)

#### 工作流程

```bash
# Step 1: 优化缺陷超胞
# Step 2: 单个原子位移并计算总能量
# Step 3: 拟合势能面获得力常数
```

#### Python 分析脚本

```python
#!/usr/bin/env python3
"""
defect_phonon_analysis.py - 缺陷局域声子分析
"""
import numpy as np
from ase.io import read
from ase.phonons import Phonons
from ase.calculators.vasp import Vasp

# 读取优化后的缺陷超胞
atoms = read('defect_opt/POSCAR')

# 设置计算器
calc = Vasp(
    xc='PBE',
    encut=520,
    kpts=(3,3,3),
    isym=0,
    ibrion=6,
    nfree=2,
    potim=0.015,
    ediff=1e-8
)
atoms.calc = calc

# 计算声子 (仅缺陷近邻区域)
# 使用超胞降低计算成本
ph = Phonons(atoms, calc, supercell=(3,3,3), delta=0.01)
ph.run()

# 获得力常数
ph.read(acoustic=False)
ph.clean()

# 计算DOS
omega = np.linspace(0, 50, 500)  # THz
dos = ph.get_dos(kpts=(10,10,10)).sample_grid(omega, sigma=0.5)

# 识别局域模式
print("分析缺陷局域声子模式...")
band_path = atoms.cell.bandpath('GXMGRX', npoints=100)
band_energies = ph.band_structure(band_path)

# 保存结果
np.save('defect_phonon_bands.npy', band_energies)
print("声子带结构已保存到 defect_phonon_bands.npy")
```

### 3. 投影声子DOS分析

```python
#!/usr/bin/env python3
"""
projected_phonon_dos.py - 投影声子态密度分析
识别缺陷原子对声子模式的贡献
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.dft.kpoints import get_special_points

def calculate_pdos(atoms, eigenvectors, mode_idx, defect_indices):
    """
    计算特定声子模式的投影DOS
    
    Args:
        atoms: ASE atoms对象
        eigenvectors: 声子本征矢量
        mode_idx: 模式索引
        defect_indices: 缺陷区域原子索引列表
    """
    # 提取模式的本征矢量
    ev = eigenvectors[mode_idx]
    
    # 计算各原子的参与率
    participation = np.abs(ev)**2
    
    # 缺陷区域贡献
    defect_contribution = np.sum(participation[defect_indices])
    total_participation = np.sum(participation)
    
    localization_ratio = defect_contribution / total_participation
    
    return localization_ratio

def identify_localized_modes(frequencies, eigenvectors, defect_indices, threshold=0.6):
    """
    识别高度局域化的声子模式
    
    Args:
        threshold: 局域化阈值 (缺陷区域贡献 > 60%)
    """
    localized_modes = []
    
    for i, freq in enumerate(frequencies):
        ratio = calculate_pdos(None, eigenvectors, i, defect_indices)
        if ratio > threshold:
            localized_modes.append({
                'index': i,
                'frequency': freq,
                'localization_ratio': ratio
            })
    
    return localized_modes

# 使用示例
if __name__ == '__main__':
    # 假设已有声子计算结果
    frequencies = np.load('frequencies.npy')
    eigenvectors = np.load('eigenvectors.npy')
    
    # 定义缺陷区域 (例如: 空位周围第一近邻)
    defect_indices = [0, 1, 2, 3, 4, 5]  # 根据具体体系调整
    
    # 识别局域模式
    modes = identify_localized_modes(frequencies, eigenvectors, defect_indices)
    
    print("缺陷局域声子模式:")
    for mode in modes:
        print(f"  模式 {mode['index']}: "
              f"{mode['frequency']:.2f} THz, "
              f"局域化率: {mode['localization_ratio']:.2%}")
```

---

## 振动熵计算

### 准谐近似 (QHA)

```python
#!/usr/bin/env python3
"""
vibrational_entropy.py - 缺陷振动熵计算
"""
import numpy as np
from scipy import integrate

def vibrational_entropy(omega, T):
    """
    计算振动熵 (单位: k_B/模式)
    
    Args:
        omega: 声子频率 (THz)
        T: 温度 (K)
    """
    hbar = 6.626e-34 / (2 * np.pi)  # J·s
    k_B = 1.381e-23                  # J/K
    THz_to_Hz = 1e12
    
    # 避免零频发散
    omega = np.maximum(omega, 0.01)  # 最小截断
    
    x = hbar * omega * THz_to_Hz / (k_B * T)
    
    # S = k_B * [x/(exp(x)-1) - ln(1-exp(-x))]
    S = k_B * (x / (np.exp(x) - 1) - np.log(1 - np.exp(-x)))
    
    return S / k_B  # 返回以k_B为单位

def defect_formation_entropy(omega_perfect, omega_defect, T):
    """
    计算缺陷形成熵 ΔS = S_defect - S_perfect
    
    注意: 需要确保模式数匹配 (使用相同超胞)
    """
    S_perfect = np.sum(vibrational_entropy(omega_perfect, T))
    S_defect = np.sum(vibrational_entropy(omega_defect, T))
    
    return S_defect - S_perfect

# 计算示例
T_range = np.linspace(100, 1000, 50)  # 100-1000K

# 假设已有完美晶体和缺陷的声子频率
omega_perfect = np.load('perfect_phonon_frequencies.npy')
omega_defect = np.load('defect_phonon_frequencies.npy')

delta_S = [defect_formation_entropy(omega_perfect, omega_defect, T) 
           for T in T_range]

# 绘制结果
import matplotlib.pyplot as plt
plt.plot(T_range, delta_S, 'b-', linewidth=2)
plt.xlabel('Temperature (K)')
plt.ylabel('ΔS (k_B)')
plt.title('Defect Formation Entropy vs Temperature')
plt.grid(True)
plt.savefig('vibrational_entropy.png', dpi=300)
```

---

## 声子对缺陷稳定性的影响

### 温度相关的缺陷形成能

```
E_f(T) = [E_defect - E_perfect - Σμ_i + q·E_F] + ΔE_ZPE - T·ΔS_vib

ΔE_ZPE = ½·Σ_λ ℏ(ω_λ^defect - ω_λ^perfect)
```

### 计算步骤

```bash
# 1. 计算完美晶体声子
# perfect_phonon/
cd perfect_phonon
phonopy -d --dim="3 3 3"
... (DFT计算力) ...
phonopy --fc vasprun.xml-001
phonopy -p band.conf

# 2. 计算缺陷体系声子
# defect_phonon/
cd defect_phonon
phonopy -d --dim="3 3 3"
# 注意: 使用相同超胞大小
... (DFT计算力) ...
phonopy --fc vasprun.xml-001

# 3. Python分析
python calculate_formation_energy_with_vibrational.py
```

```python
#!/usr/bin/env python3
"""
formation_energy_phonon.py - 包含声子贡献的缺陷形成能
"""
import numpy as np
import yaml

def load_phonopy_yaml(filename):
    """读取phonopy输出"""
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    
    frequencies = []
    for band in data['phonon']:
        for mode in band['band']:
            frequencies.append(mode['frequency'])
    
    return np.array(frequencies)  # THz

def zero_point_energy(frequencies):
    """计算零点能 (eV)"""
    hbar = 6.626e-34 / (2 * np.pi)  # J·s
    THz_to_Hz = 1e12
    eV_to_J = 1.602e-19
    
    # ZPE = 0.5 * sum(hbar * omega)
    zpe = 0.5 * np.sum(hbar * frequencies * THz_to_Hz) / eV_to_J
    return zpe

def vibrational_free_energy(frequencies, T):
    """计算振动自由能 (eV)"""
    hbar = 6.626e-34 / (2 * np.pi)
    k_B = 1.381e-23
    THz_to_Hz = 1e12
    eV_to_J = 1.602e-19
    
    x = hbar * frequencies * THz_to_Hz / (k_B * T)
    
    # F_vib = k_B*T * sum(ln(2*sinh(x/2)))
    f_vib = k_B * T * np.sum(np.log(2 * np.sinh(x/2))) / eV_to_J
    
    return f_vib

# 主计算
def calculate_formation_energy_with_phonon(E_form_0K, T, 
                                           phonon_perfect, phonon_defect):
    """
    计算包含声子贡献的形成能
    
    Args:
        E_form_0K: 0K静态形成能 (eV)
        T: 温度 (K)
        phonon_perfect: 完美晶体声子文件
        phonon_defect: 缺陷声子文件
    """
    # 加载声子数据
    freq_p = load_phonopy_yaml(phonon_perfect)
    freq_d = load_phonopy_yaml(phonon_defect)
    
    # 零点能修正
    zpe_p = zero_point_energy(freq_p)
    zpe_d = zero_point_energy(freq_d)
    delta_zpe = zpe_d - zpe_p
    
    # 热贡献
    f_vib_p = vibrational_free_energy(freq_p, T)
    f_vib_d = vibrational_free_energy(freq_d, T)
    delta_f_vib = f_vib_d - f_vib_p
    
    # 总形成能
    E_form_T = E_form_0K + delta_zpe + delta_f_vib
    
    return {
        'E_form_0K': E_form_0K,
        'delta_ZPE': delta_zpe,
        'delta_F_vib': delta_f_vib,
        'E_form_T': E_form_T
    }

# 示例使用
if __name__ == '__main__':
    E_form_0K = 3.5  # eV, 来自静态计算
    
    for T in [300, 600, 900, 1200]:
        result = calculate_formation_energy_with_phonon(
            E_form_0K, T,
            'perfect/phonopy.yaml',
            'defect/phonopy.yaml'
        )
        
        print(f"\nT = {T}K:")
        print(f"  0K形成能: {result['E_form_0K']:.3f} eV")
        print(f"  ZPE修正: {result['delta_ZPE']:+.3f} eV")
        print(f"  振动自由能修正: {result['delta_F_vib']:+.3f} eV")
        print(f"  总形成能: {result['E_form_T']:.3f} eV")
```

---

## 非谐效应

### 温度依赖的声子重整化

高温下需要考虑声子-声子相互作用：

```
ω(T) = ω₀ + Δω(T)

Δω(T): 自能修正 (需要计算三阶力常数)
```

### 第三方软件

| 软件 | 功能 | 接口 |
|------|------|------|
| **Phonopy+Phono3py** | 三阶力常数/热导 | VASP/QE |
| **TDEP** | 从头算分子动力学声子 | 多种DFT |
| **ALAMODE** | 非谐晶格动力学 | VASP/QE |

---

## 实际案例: Si空位

### 计算设置

```bash
# 超胞: 3×3×3 (216原子)
# 空位位置: 中心
# 计算类型: VASP finite difference
```

### 关键结果

| 性质 | 完美Si | Si空位 | 变化 |
|------|--------|--------|------|
| 低频声子 | 标准声学支 | 新增~2 THz共振模 | - |
| 高频光学支 | 15-18 THz | 出现19.5 THz局域模 | 蓝移 |
| ZPE贡献 | 参考 | +0.08 eV | 空位更易形成 |
| ΔS_vib@1000K | 参考 | +2.3 k_B | 熵稳定化 |

---

## 故障排查

### 问题1: 虚频出现

**原因**: 
- 结构未完全优化
- 超胞太小

**解决**:
```bash
# 重新优化，更严格标准
EDIFFG = -0.001      # 更小的力标准
NSW = 200
```

### 问题2: 模式数不匹配

**原因**: 完美晶体和缺陷超胞大小不同

**解决**:
```bash
# 使用相同超胞
# 完美晶体计算时也使用含空位的超胞
# 但不放入空位 (即有一个"假原子")
```

---

## 参考资源

- [Phonopy缺陷声子文档](https://phonopy.github.io/phonopy/)
- [TDEP非谐晶格动力学](https://github.com/tdep-developers/tdep)
- [ALAMODE非谐计算](http://alamode.sourceforge.net/)
