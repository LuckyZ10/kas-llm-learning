# Dzyaloshinskii-Moriya 相互作用 (DMI)

## 概述

Dzyaloshinskii-Moriya相互作用（DMI）是一种反对称交换相互作用，在缺乏中心对称性的体系中产生。它是斯格明子（skyrmion）等拓扑磁性结构形成的关键机制，在自旋电子学器件中有重要应用。

---

## 理论基础

### DMI 哈密顿量

```
H_DMI = Σ_ij D_ij · (S_i × S_j)
```

其中：
- **D_ij**: DMI矢量 (反对称: D_ij = -D_ji)
- **S_i, S_j**: 相邻自旋
- **×**: 矢量叉乘

### 微观起源

DMI来源于自旋-轨道耦合（SOC）与交换相互作用的共同作用：

```
D_ij ∝ λ_soc · (r_ij × R_ij) / |r_ij|³
```

- **λ_soc**: 自旋-轨道耦合强度
- **r_ij**: 原子间位移矢量
- **R_ij**: 配体位置矢量

### DMI 分类

#### 块体DMI (Bulk DMI)
- 存在于非中心对称晶体中（如B20结构：MnSi, FeGe）
- 形成螺旋磁结构

#### 界面DMI (Interfacial DMI)
- 存在于界面/表面（如Co/Pt, Co/Ir）
- 形成奈尔型斯格明子
- DMI矢量垂直于界面

---

## 计算方法

### 1. 四态能量法 (Four-State Energy Method)

#### 原理

通过计算四种共线自旋构型的能量来提取交换参数和DMI：

```
J_ij = (E↑↑ + E↓↓ - E↑↓ - E↓↑) / 4
D_z = (E↑→ + E↓← - E↑← - E↓→) / 4   (对于z方向DMI)
```

#### VASP 设置

```bash
# INCAR.dmi_calculation
ISPIN = 2
LSORBIT = .TRUE.        # 开启SOC (必需)
SAXIS = 0 0 1           # 初始自旋方向

# 固定自旋方向计算
IWAVPR = 11             # 从WAVECAR开始
NELM = 1                # 仅1步电子迭代 (固定密度)

# 高精度
EDIFF = 1E-8
ENCUT = 600
ISMEAR = 0
SIGMA = 0.01
```

#### 四态构型设置

```bash
# 态1: ↑↑ (两原子自旋均向上)
# MAGMOM = 2*2 0 0  2*2 0 0

# 态2: ↓↓ (两原子自旋均向下)
# MAGMOM = -2*2 0 0  -2*2 0 0

# 态3: ↑↓ (原子1向上，原子2向下)
# MAGMOM = 2*2 0 0  -2*2 0 0

# 态4: ↓↑ (原子1向下，原子2向上)
# MAGMOM = -2*2 0 0  2*2 0 0

# 态5: ↑→ (用于DMI_y)
# MAGMOM = 0 2*2 0  2*2 0 0

# 态6: ↑← (用于DMI_y)
# MAGMOM = 0 -2*2 0  2*2 0 0
```

#### Python 实现

```python
#!/usr/bin/env python3
"""
dmi_four_state_method.py - 四态法计算DMI
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SpinConfiguration:
    """自旋构型"""
    name: str
    S1: np.ndarray  # 原子1自旋方向
    S2: np.ndarray  # 原子2自旋方向
    energy: float = None

class DMICalculator:
    """DMI计算器 - 四态法"""
    
    def __init__(self):
        self.configurations = []
    
    def setup_collinear_states(self):
        """设置四态共线构型"""
        states = [
            SpinConfiguration("↑↑", np.array([0, 0, 1]), np.array([0, 0, 1])),
            SpinConfiguration("↓↓", np.array([0, 0, -1]), np.array([0, 0, -1])),
            SpinConfiguration("↑↓", np.array([0, 0, 1]), np.array([0, 0, -1])),
            SpinConfiguration("↓↑", np.array([0, 0, -1]), np.array([0, 0, 1])),
        ]
        return states
    
    def setup_noncollinear_states(self):
        """设置非共线构型用于DMI提取"""
        states = [
            # x方向DMI
            SpinConfiguration("↑→", np.array([0, 0, 1]), np.array([1, 0, 0])),
            SpinConfiguration("↑←", np.array([0, 0, 1]), np.array([-1, 0, 0])),
            SpinConfiguration("↓→", np.array([0, 0, -1]), np.array([1, 0, 0])),
            SpinConfiguration("↓←", np.array([0, 0, -1]), np.array([-1, 0, 0])),
            
            # y方向DMI
            SpinConfiguration("↑↗", np.array([0, 0, 1]), np.array([0, 1, 0])),
            SpinConfiguration("↑↙", np.array([0, 0, 1]), np.array([0, -1, 0])),
            SpinConfiguration("↓↗", np.array([0, 0, -1]), np.array([0, 1, 0])),
            SpinConfiguration("↓↙", np.array([0, 0, -1]), np.array([0, -1, 0])),
        ]
        return states
    
    def calculate_exchange_parameters(self, energies: dict) -> dict:
        """
        从四态能量计算交换参数
        
        Args:
            energies: {'↑↑': E1, '↓↓': E2, '↑↓': E3, '↓↑': E4}
        
        Returns:
            {'J': 交换积分, 'D': DMI矢量}
        """
        E_uu = energies['↑↑']
        E_dd = energies['↓↓']
        E_ud = energies['↑↓']
        E_du = energies['↓↑']
        
        # Heisenberg交换
        J = (E_uu + E_dd - E_ud - E_du) / 4
        
        # DMI z分量 (从↑→, ↑←, ↓→, ↓←构型)
        if '↑→' in energies:
            E_ur = energies['↑→']
            E_ul = energies['↑←']
            E_dr = energies['↓→']
            E_dl = energies['↓←']
            
            D_z = (E_ur + E_dl - E_ul - E_dr) / 4
        else:
            D_z = 0
        
        # DMI y分量 (从↗↙构型)
        if '↑↗' in energies:
            E_uu_y = energies['↑↗']
            E_ud_y = energies['↑↙']
            E_du_y = energies['↓↗']
            E_dd_y = energies['↓↙']
            
            D_y = (E_uu_y + E_dd_y - E_ud_y - E_du_y) / 4
        else:
            D_y = 0
        
        return {
            'J': J,
            'D': np.array([0, D_y, D_z]),  # 假设D_x = 0 (对称性)
            'D_magnitude': np.sqrt(D_y**2 + D_z**2)
        }
    
    def generate_vasp_magmom(self, config: SpinConfiguration, 
                            magnetic_moment: float = 2.0) -> str:
        """
        生成VASP MAGMOM设置
        
        Args:
            magnetic_moment: 磁矩大小 (μB)
        """
        S1 = config.S1 * magnetic_moment
        S2 = config.S2 * magnetic_moment
        
        magmom_str = f"{S1[0]:.3f} {S1[1]:.3f} {S1[2]:.3f} "
        magmom_str += f"{S2[0]:.3f} {S2[1]:.3f} {S2[2]:.3f}"
        
        return magmom_str

def create_vasp_inputs_for_dmi(output_dir: str = 'dmi_calculations'):
    """
    生成四态法VASP输入文件
    """
    import os
    
    calculator = DMICalculator()
    states = calculator.setup_collinear_states()
    states.extend(calculator.setup_noncollinear_states())
    
    # 基础INCAR模板
    incar_template = """DMI Calculation - Four State Method
   ISPIN = 2
   LSORBIT = .TRUE.
   SAXIS = 0 0 1
   
   ENCUT = 600
   EDIFF = 1E-8
   ISMEAR = 0
   SIGMA = 0.01
   
   NELM = 100
   NELMIN = 6
   
   ISYM = 0
   IWAVPR = 11
"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for state in states:
        state_dir = os.path.join(output_dir, state.name)
        os.makedirs(state_dir, exist_ok=True)
        
        # 写入INCAR
        incar = incar_template + f"\n   MAGMOM = {calculator.generate_vasp_magmom(state)}\n"
        with open(os.path.join(state_dir, 'INCAR'), 'w') as f:
            f.write(incar)
        
        print(f"Created {state_dir}/INCAR")
        print(f"  MAGMOM: {calculator.generate_vasp_magmom(state)}")

if __name__ == '__main__':
    # 示例使用
    create_vasp_inputs_for_dmi()
    
    # 假设已有能量数据
    energies = {
        '↑↑': -100.50,
        '↓↓': -100.48,
        '↑↓': -100.20,
        '↓↑': -100.22,
        '↑→': -100.35,
        '↑←': -100.25,
        '↓→': -100.15,
        '↓←': -100.25,
    }
    
    calc = DMICalculator()
    params = calc.calculate_exchange_parameters(energies)
    
    print("\n计算结果:")
    print(f"  交换积分 J: {params['J']:.4f} meV")
    print(f"  DMI矢量 D: [{params['D'][0]:.4f}, {params['D'][1]:.4f}, {params['D'][2]:.4f}] meV")
    print(f"  DMI大小 |D|: {params['D_magnitude']:.4f} meV")
    print(f"  D/J 比率: {params['D_magnitude']/abs(params['J']):.4f}")
```

### 2. 线性响应理论 (Torque Method)

```python
#!/usr/bin/env python3
"""
dmi_torque_method.py - 扭矩法计算DMI
基于线性响应理论，通过计算自旋扭矩来提取DMI
"""
import numpy as np

class TorqueMethodDMI:
    """
    扭矩法计算DMI
    
    参考: Xiang et al., PRB 84, 054419 (2011)
    """
    
    def __init__(self, structure):
        self.structure = structure
        self.magnetic_atoms = []  # 磁性原子索引
    
    def calculate_constrained_torque(self, spin_direction: np.ndarray,
                                    constraint_strength: float = 1.0) -> np.ndarray:
        """
        计算约束自旋方向时的扭矩
        
        Args:
            spin_direction: 约束自旋方向 [Sx, Sy, Sz]
            constraint_strength: 约束势强度
        
        Returns:
            torque: 扭矩矢量 [Tx, Ty, Tz]
        """
        # 这需要从VASP计算输出中提取
        # 使用LSDA+U+SOC计算
        
        # 简化的扭矩计算公式
        # T = -∂E/∂θ (极角)
        
        torque = np.zeros(3)
        
        # 从OUTCAR提取数据
        # 实际实现需要解析VASP输出
        
        return torque
    
    def extract_dmi_from_torque(self, torques: dict) -> np.ndarray:
        """
        从不同自旋方向的扭矩提取DMI
        
        Args:
            torques: {direction: torque_vector}
        
        Returns:
            DMI_vector: D_ij
        """
        # DMI与扭矩的关系
        # τ_i = Σ_j D_ij × S_j
        
        # 通过多个方向的扭矩求解DMI
        DMI = np.zeros(3)
        
        # 线性方程组求解
        # [τ] = [D] × [S]
        
        return DMI

# VASP约束磁性计算设置
def get_constrained_moment_incar():
    """
    VASP约束磁矩计算INCAR
    
    使用LAMBDA参数约束自旋方向
    """
    incar = """
# Constrained moment calculation
   I_CONSTRAINED_M = 1    # 开启约束
   LAMBDA = 10.0          # 约束强度
   
   ISPIN = 2
   LSORBIT = .TRUE.
   
# M_CONSTR 指定约束方向 (在MAGMOM中)
   MAGMOM = 2*3 0 0 1    # 3原子，初始沿z方向
   M_CONSTR = 0 0 1      # 约束到z方向
"""
    return incar
```

### 3. Wannier函数投影法

```python
#!/usr/bin/env python3
"""
dmi_wannier_projection.py - 基于Wannier函数的DMI计算
"""
import numpy as np

def calculate_dmi_from_wannier(hr_file: str, atom_pairs: list):
    """
    从Wannier90 HR.dat计算DMI
    
    Args:
        hr_file: Wannier90_hr.dat文件路径
        atom_pairs: 原子对列表 [(i,j), ...]
    
    Returns:
        DMI矩阵
    """
    # 读取Wannier哈密顿量
    with open(hr_file, 'r') as f:
        lines = f.readlines()
    
    # 解析HR数据
    # H(R)_{m,n} 矩阵
    
    DMI = {}
    
    for i, j in atom_pairs:
        # DMI = Im[H_up_down - H_down_up]
        # 从Wannier哈密顿提取
        
        D_ij = np.zeros(3)
        
        # 计算各分量
        # D_x ∝ Im[H_↑↓ - H_↓↑] (x方向)
        # D_y ∝ Im[H_↑↓ - H_↓↑] (y方向)
        # D_z ∝ Re[H_↑↓ - H_↓↑] (z方向)
        
        DMI[(i, j)] = D_ij
    
    return DMI
```

---

## 结果分析

### DMI 强度评估

```python
#!/usr/bin/env python3
"""
dmi_analysis.py - DMI结果分析
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def analyze_dmi_strength(J: float, D: np.ndarray) -> dict:
    """
    分析DMI相对强度
    
    Args:
        J: Heisenberg交换 (meV)
        D: DMI矢量 (meV)
    
    Returns:
        分析结果字典
    """
    D_mag = np.linalg.norm(D)
    ratio = D_mag / abs(J) if J != 0 else float('inf')
    
    # 手性判断
    chirality = "right-handed" if D[2] > 0 else "left-handed"
    
    # 磁结构预测
    if ratio < 0.1:
        predicted_structure = "ferromagnetic"
    elif ratio < 0.5:
        predicted_structure = "weak_helix"
    elif ratio < 2.0:
        predicted_structure = "strong_helix"
    else:
        predicted_structure = "skyrmion_phase"
    
    return {
        'D_magnitude': D_mag,
        'D_direction': D / D_mag if D_mag > 0 else np.zeros(3),
        'D_J_ratio': ratio,
        'chirality': chirality,
        'predicted_structure': predicted_structure,
        'helix_period': 2 * np.pi * abs(J) / D_mag if D_mag > 0 else float('inf')
    }

def visualize_dmi_on_lattice(positions: np.ndarray, 
                             DMI_vectors: dict,
                             output: str = 'dmi_visualization.png'):
    """
    可视化晶格上的DMI矢量
    
    Args:
        positions: 原子位置 (N×3数组)
        DMI_vectors: {(i,j): D_ij}
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原子位置
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
              c='blue', s=100, label='Atoms')
    
    # 绘制DMI矢量
    for (i, j), D_ij in DMI_vectors.items():
        pos_i = positions[i]
        pos_j = positions[j]
        midpoint = (pos_i + pos_j) / 2
        
        # 绘制DMI矢量
        ax.quiver(midpoint[0], midpoint[1], midpoint[2],
                 D_ij[0], D_ij[1], D_ij[2],
                 length=0.5, normalize=True, color='red', alpha=0.6)
        
        # 绘制连接
        ax.plot([pos_i[0], pos_j[0]], 
               [pos_i[1], pos_j[1]], 
               [pos_i[2], pos_j[2]], 
               'k-', alpha=0.3)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('DMI Vector Visualization')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

def compare_dmi_systems(results: dict):
    """
    比较不同体系的DMI特征
    
    Args:
        results: {'system_name': {'J': ..., 'D': ...}}
    """
    print("\nDMI比较:")
    print("-" * 70)
    print(f"{'System':<20} {'J (meV)':<12} {'|D| (meV)':<12} {'D/J':<12} {'Structure':<15}")
    print("-" * 70)
    
    for name, data in results.items():
        analysis = analyze_dmi_strength(data['J'], data['D'])
        print(f"{name:<20} {data['J']:<12.2f} "
              f"{analysis['D_magnitude']:<12.2f} "
              f"{analysis['D_J_ratio']:<12.3f} "
              f"{analysis['predicted_structure']:<15}")
```

---

## 典型体系参数

### B20化合物 (块体DMI)

| 材料 | J (meV) | D (meV) | D/J | 螺旋周期 (nm) |
|------|---------|---------|-----|--------------|
| MnSi | 1.4 | 0.58 | 0.41 | 18 |
| FeGe | 2.1 | 0.64 | 0.30 | 70 |
| FeCoSi | 3.2 | 0.98 | 0.31 | 90 |

### 重金属/铁磁体界面 (界面DMI)

| 界面 | D_s (mJ/m²) | 方向 | 技术 |
|------|------------|------|------|
| Co/Pt | 1.5-3.0 | z | 四态法 |
| Co/Ir | 2.0-4.0 | z | 扭矩法 |
| Ni/Pd | 0.5-1.0 | z | 自旋波 |

---

## 故障排查

### 问题1: DMI计算值为零

**可能原因**:
- 体系存在中心对称性
- SOC未开启
- 原子对太远

**检查**:
```python
def check_symmetry(structure):
    """检查结构对称性"""
    # 使用spglib检查中心对称
    from spglib import get_symmetry_dataset
    
    dataset = get_symmetry_dataset(structure)
    has_inversion =  any(
        np.allclose(dataset['rotations'][i], -np.eye(3))
        for i in range(len(dataset['rotations']))
    )
    
    return not has_inversion  # True表示可能有DMI
```

### 问题2: 收敛困难

**解决方案**:
```bash
# 增加混合参数
AMIX = 0.1
BMIX = 0.0001
AMIX_MAG = 0.2
BMIX_MAG = 0.0001

# 增加空态
NBANDS = 1.5 * NELECT / 2

# 更严格的收敛标准
EDIFF = 1E-9
```

---

## 参考资源

- [VASP磁性计算Wiki](https://www.vasp.at/wiki/index.php/Magnetic_calculations)
- [四态法论文](https://doi.org/10.1103/PhysRevB.91.224408)
- [扭矩法论文](https://doi.org/10.1103/PhysRevB.84.054419)
- [Wannier90文档](http://www.wannier.org/)
