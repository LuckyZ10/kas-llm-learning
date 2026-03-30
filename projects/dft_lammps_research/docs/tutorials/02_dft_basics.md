# 02 - DFT计算基础教程 | DFT Calculation Basics

> **学习目标**: 掌握VASP/Quantum ESPRESSO计算设置、收敛性测试和数据处理  
> **Learning Goal**: Master VASP/QE setup, convergence testing, and data processing

---

## 📋 目录 | Table of Contents

1. [理论基础 | Theoretical Background](#1-理论基础--theoretical-background)
2. [输入文件准备 | Input File Preparation](#2-输入文件准备--input-file-preparation)
3. [收敛性测试 | Convergence Testing](#3-收敛性测试--convergence-testing)
4. [计算流程 | Calculation Workflow](#4-计算流程--calculation-workflow)
5. [结果解析 | Results Analysis](#5-结果解析--results-analysis)
6. [常见错误与解决 | Common Errors & Solutions](#6-常见错误与解决--common-errors--solutions)
7. [练习题 | Exercises](#7-练习题--exercises)

---

## 1. 理论基础 | Theoretical Background

### 1.1 Kohn-Sham方程 | The Kohn-Sham Equations

密度泛函理论(DFT)基于Hohenberg-Kohn定理，通过Kohn-Sham方程求解：

$$
\left[-\frac{\hbar^2}{2m}\nabla^2 + V_{eff}(\mathbf{r})\right]\psi_i(\mathbf{r}) = \varepsilon_i\psi_i(\mathbf{r})
$$

其中有效势：

$$
V_{eff}(\mathbf{r}) = V_{ext}(\mathbf{r}) + \int \frac{n(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}d\mathbf{r}' + V_{xc}[n(\mathbf{r})]
$$

### 1.2 交换关联泛函 | Exchange-Correlation Functionals

| 泛函 | Functional | 类型 | 适用场景 |
|------|-----------|------|----------|
| LDA | 局域密度近似 | 快速估算、初步筛选 |
| PBE | GGA | 平衡精度与速度 |
| PBEsol | GGA | 固体、表面计算 |
| SCAN | meta-GGA | 高精度计算 |
| HSE06 | 杂化泛函 | 带隙计算 |

### 1.3 赝势方法 | Pseudopotential Methods

```
┌─────────────────────────────────────────────────────────────┐
│                    全电子波函数                              │
│                Core: 1s² 2s² 2p⁶ (Os,快速振荡)              │
│                Valence: 3s² 3p⁶ 3d¹⁰ 4s² (缓慢变化)         │
└─────────────────────────────────────────────────────────────┘
                              ↓
                        赝势近似
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    赝波函数                                  │
│                芯态: 被赝势替代                              │
│                价态: 与全电子在截断半径外一致                │
└─────────────────────────────────────────────────────────────┘
```

**常用赝势库**: PAW (VASP), PSLibrary (QE), ONCV (通用)

---

## 2. 输入文件准备 | Input File Preparation

### 2.1 VASP输入文件 | VASP Input Files

#### INCAR - 主控制文件

```fortran
# 基础设置 | Basic Settings
SYSTEM = Li3PS4 Structure Optimization
ISTART = 0          # 从头计算
ICHARG = 2          # 从原子密度叠加
PREC = Accurate     # 精度级别

# 电子步设置 | Electronic Settings
ENCUT = 520         # 截断能 (eV)，需测试收敛性
EDIFF = 1E-6        # 电子收敛标准 (eV)
ISMEAR = 0          # Gaussian展宽
SIGMA = 0.05        # 展宽宽度 (eV)
ALGO = Fast         # RMM-DIIS算法
NELMIN = 4          # 最小电子步数
NELM = 100          # 最大电子步数

# 离子步设置 | Ionic Settings
IBRION = 2          # 共轭梯度优化
ISIF = 3            # 优化晶胞+原子位置
NSW = 200           # 最大离子步数
EDIFFG = -0.01      # 力收敛标准 (eV/Å)

# 写入控制 | Write Control
LWAVE = .FALSE.     # 不保存WAVECAR
LCHARG = .TRUE.     # 保存CHGCAR
LVTOT = .FALSE.

# 并行设置 | Parallel Settings
NCORE = 4           # 每节点核心数
LREAL = Auto        # 自动投影

# 其他 | Others
NELMIN = 4          # 最小电子步
NELM = 100          # 最大电子步
```

#### POSCAR - 结构文件

```
Li3PS4                      # 系统名称
1.0                         # 晶格缩放因子
6.088900 0.000000 0.000000  # 晶格向量a
0.000000 5.272000 0.000000  # 晶格向量b
0.000000 0.000000 6.093700  # 晶格向量c
Li P S                      # 元素类型
12 4 16                     # 各元素原子数
Direct                      # 坐标类型:分数坐标
0.250000 0.750000 0.250000  # Li原子位置
0.750000 0.250000 0.750000
...
```

#### KPOINTS - k点网格

```
Automatic mesh            # 自动生成网格
0                         # 自动模式
Gamma                     # Gamma-centered
5 5 5                     # k点网格 (需收敛测试)
0 0 0                     # 网格偏移
```

#### POTCAR - 赝势文件

```bash
# 使用pymatgen生成 | Generate with pymatgen
from pymatgen.io.vasp import Potcar
potcar = Potcar(symbols=['Li', 'P', 'S'], functional='PBE')
potcar.write_file('POTCAR')
```

### 2.2 Quantum ESPRESSO输入 | QE Input

#### pw.x输入文件 (vc-relax)

```fortran
&CONTROL
  calculation = 'vc-relax'
  restart_mode = 'from_scratch'
  prefix = 'Li3PS4'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  tprnfor = .true.
  tstress = .true.
/
&SYSTEM
  ibrav = 0
  nat = 32
  ntyp = 3
  ecutwfc = 40          ! 波函数截断能 (Ry)
  ecutrho = 320         ! 电荷密度截断能 (Ry)
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.7
  diagonalization = 'david'
/
&IONS
  ion_dynamics = 'bfgs'
/
&CELL
  cell_dynamics = 'bfgs'
  press = 0.0
  press_conv_thr = 0.5
/
ATOMIC_SPECIES
Li 6.94 Li.pbe-sl-rrkjus_psl.1.0.0.UPF
P 30.97 P.pbe-n-rrkjus_psl.1.0.0.UPF
S 32.06 S.pbe-nl-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS {crystal}
Li 0.250000 0.750000 0.250000
...
K_POINTS {automatic}
5 5 5 0 0 0              ! k点网格
```

### 2.3 使用Python生成输入 | Generate Input with Python

```python
"""
DFT输入文件生成器 | DFT Input Generator
"""
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.calculators.espresso import Espresso
from ase.io import read

# 读取结构 | Read structure
atoms = read('Li3PS4.vasp')

# ========== VASP计算器设置 ==========
vasp_calc = Vasp(
    # 基本设置
    xc='PBE',
    encut=520,
    prec='Accurate',
    
    # k点网格
    kpts=(5, 5, 5),           # 或 kpts={'density': 0.25}
    
    # 电子步
    ediff=1e-6,
    ismear=0,
    sigma=0.05,
    algo='Fast',
    
    # 离子步
    ibrion=2,
    isif=3,
    nsw=200,
    ediffg=-0.01,
    
    # 输出控制
    lwave=False,
    lcharg=True,
    
    # 并行
    ncore=4,
    
    # 命令
    command='mpirun -np 4 vasp_std'
)

atoms.calc = vasp_calc

# 运行计算 | Run calculation
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

# ========== QE计算器设置 ==========
pseudopotentials = {
    'Li': 'Li.pbe-sl-rrkjus_psl.1.0.0.UPF',
    'P': 'P.pbe-n-rrkjus_psl.1.0.0.UPF',
    'S': 'S.pbe-nl-rrkjus_psl.1.0.0.UPF'
}

qe_calc = Espresso(
    pseudopotentials=pseudopotentials,
    pseudo_dir='./pseudo',
    input_data={
        'control': {
            'calculation': 'vc-relax',
            'prefix': 'Li3PS4',
            'outdir': './tmp',
        },
        'system': {
            'ecutwfc': 40,
            'ecutrho': 320,
            'occupations': 'smearing',
            'smearing': 'gaussian',
            'degauss': 0.01,
        },
        'electrons': {
            'conv_thr': 1e-8,
            'mixing_beta': 0.7,
        },
        'ions': {
            'ion_dynamics': 'bfgs',
        },
        'cell': {
            'cell_dynamics': 'bfgs',
        }
    },
    kpts=(5, 5, 5),
    command='mpirun -np 4 pw.x -in PREFIX.pwi > PREFIX.pwo'
)

atoms.calc = qe_calc
```

---

## 3. 收敛性测试 | Convergence Testing

### 3.1 截断能收敛测试 | ENCUT Convergence

```python
"""
截断能收敛测试 | ENCUT Convergence Test
"""
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.calculators.vasp import Vasp

def test_encut_convergence(atoms, encut_range, output_dir='./conv_test'):
    """
    测试截断能收敛性
    
    Args:
        atoms: ASE Atoms对象
        encut_range: 截断能列表 (eV)
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for encut in encut_range:
        print(f"Testing ENCUT = {encut} eV...")
        
        # 设置计算器
        calc = Vasp(
            xc='PBE',
            encut=encut,
            kpts=(4, 4, 4),  # 固定k点
            ibrion=-1,       # 单点计算
            nsw=0,
            command='mpirun -np 4 vasp_std'
        )
        
        atoms.calc = calc
        
        try:
            energy = atoms.get_potential_energy()
            results.append({'encut': encut, 'energy': energy})
            print(f"  Energy: {energy:.6f} eV")
        except Exception as e:
            print(f"  Failed: {e}")
    
    # 分析收敛性
    energies = np.array([r['energy'] for r in results])
    encuts = np.array([r['encut'] for r in results])
    
    # 计算能量差
    energy_diff = np.abs(np.diff(energies))
    
    print("\n" + "="*60)
    print("收敛测试结果 | Convergence Test Results:")
    print("="*60)
    for i, (encut, diff) in enumerate(zip(encuts[1:], energy_diff)):
        status = "✓ Converged" if diff < 1e-3 else "✗ Not converged"
        print(f"ENCUT={encut:4d} eV: ΔE = {diff:.6f} eV {status}")
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(encuts, energies, 'bo-')
    ax1.set_xlabel('ENCUT (eV)')
    ax1.set_ylabel('Energy (eV)')
    ax1.set_title('Total Energy vs ENCUT')
    ax1.grid(True)
    
    ax2.semilogy(encuts[1:], energy_diff, 'ro-')
    ax2.axhline(y=1e-3, color='g', linestyle='--', label='Convergence threshold')
    ax2.set_xlabel('ENCUT (eV)')
    ax2.set_ylabel('|ΔE| (eV)')
    ax2.set_title('Energy Difference vs ENCUT')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/encut_convergence.png', dpi=150)
    plt.close()
    
    return results

# 运行测试
from ase.io import read
atoms = read('Li3PS4.vasp')

encut_range = [300, 350, 400, 450, 500, 550, 600, 700]
results = test_encut_convergence(atoms, encut_range)
```

**预期输出**:
```
Testing ENCUT = 300 eV...
  Energy: -142.345123 eV
Testing ENCUT = 350 eV...
  Energy: -143.892456 eV
...
============================================================
收敛测试结果 | Convergence Test Results:
============================================================
ENCUT= 350 eV: ΔE = 1.547333 eV ✗ Not converged
ENCUT= 400 eV: ΔE = 0.123456 eV ✗ Not converged
ENCUT= 450 eV: ΔE = 0.023456 eV ✗ Not converged
ENCUT= 500 eV: ΔE = 0.001234 eV ✓ Converged
ENCUT= 550 eV: ΔE = 0.000234 eV ✓ Converged
```

### 3.2 k点网格收敛测试 | K-point Convergence

```python
def test_kpoint_convergence(atoms, kpoint_grids, encut=520):
    """k点网格收敛测试 | K-point convergence test"""
    results = []
    
    for k in kpoint_grids:
        print(f"Testing k-grid = {k}...")
        
        calc = Vasp(
            xc='PBE',
            encut=encut,
            kpts=k,
            ibrion=-1,
            nsw=0,
            command='mpirun -np 4 vasp_std'
        )
        
        atoms.calc = calc
        
        try:
            energy = atoms.get_potential_energy()
            results.append({'kpts': k, 'energy': energy})
            print(f"  Energy: {energy:.6f} eV")
        except Exception as e:
            print(f"  Failed: {e}")
    
    return results

# 测试k点
kpoint_grids = [(2,2,2), (3,3,3), (4,4,4), (5,5,5), (6,6,6), (8,8,8)]
results = test_kpoint_convergence(atoms, kpoint_grids)
```

### 3.3 收敛性测试检查表 | Convergence Checklist

| 参数 | Parameter | 测试范围 | Test Range | 收敛标准 | Convergence Criterion |
|------|-----------|---------|------------|---------|----------------------|
| ENCUT | 300-700 eV | 500 eV | ΔE < 1 meV/atom |
| k-points | 2×2×2 到 8×8×8 | 5×5×5 | ΔE < 1 meV/atom |
| SIGMA | 0.01-0.2 eV | 0.05 eV | 熵 < 1 meV/atom |
| 真空层 | 10-25 Å | 15 Å | 能量变化 < 1 meV |

---

## 4. 计算流程 | Calculation Workflow

### 4.1 标准计算流程 | Standard Calculation Flow

```
1. 结构优化 (ISIF=3)
   └── 优化晶胞和原子位置
   └── Optimize cell and atomic positions
   
2. 静态计算 (IBRION=-1)
   └── 高精度单点能
   └── High-precision single point
   
3. 电子性质计算 (Optional)
   ├── DOS计算 (LORBIT=11)
   ├── 能带计算 (ICHARG=11)
   └── ELF/电荷密度 (LELF=.TRUE.)
   
4. 振动性质计算 (Optional)
   └── 声子计算 (IBRION=5,6,7,8)
   └── Phonon calculation
```

### 4.2 不同类型的计算 | Calculation Types

```python
# ========== 结构优化 | Structure Optimization ==========
calc_relax = Vasp(
    ibrion=2,      # 共轭梯度
    isif=3,        # 优化cell+positions
    nsw=200,       # 最大步数
    ediffg=-0.01,  # 力收敛标准
)

# ========== 静态计算 | Static Calculation ==========
calc_static = Vasp(
    ibrion=-1,     # 不移动原子
    nsw=0,         # 无离子步
    nelm=200,      # 更多电子步
    ediff=1e-7,    # 更严格的电子收敛
)

# ========== DOS计算 | DOS Calculation ==========
calc_dos = Vasp(
    ibrion=-1,
    ismear=-5,     # Tetrahedron方法
    nedos=5001,    # DOS点数
    lorbit=11,     # 投影DOS
    lcharg=.TRUE.,
    lwave=.TRUE.,  # 需要WAVECAR
)

# ========== 能带计算 | Band Structure ==========
calc_band = Vasp(
    ibrion=-1,
    icharg=11,     # 从CHGCAR读取
    ismear=0,
    sigma=0.05,
    lwave=.FALSE.,
    lcharg=.FALSE.,
    kpts={'path': 'GXMGRX', 'npoints': 100},  # 特殊k点路径
)

# ========== AIMD计算 | AIMD Calculation ==========
calc_aimd = Vasp(
    ibrion=0,      # MD模拟
    mdalgo=2,      # Nose-Hoover
    smass=0,       # Nose质量参数
    tebeg=300,     # 起始温度
    teend=300,     # 结束温度
    potim=1.0,     # 时间步长(fs)
    nsw=10000,     # MD步数
    ismear=0,
    sigma=0.1,     # MD需要更大展宽
)
```

---

## 5. 结果解析 | Results Analysis

### 5.1 OUTCAR解析 | Parsing OUTCAR

```python
"""
VASP OUTCAR解析工具 | VASP OUTCAR Parser
"""
from ase.io import read
from ase.io.vasp import read_vasp_out
import numpy as np

# ========== 使用ASE解析 | Parse with ASE ==========
# 读取所有离子步 | Read all ionic steps
frames = read('OUTCAR', index=':', format='vasp-out')

# 读取单帧 | Read single frame
atoms = read('OUTCAR', index=-1, format='vasp-out')  # 最后一帧

# 获取能量和力 | Get energy and forces
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()  # Voigt notation

print(f"能量 | Energy: {energy:.6f} eV")
print(f"能量/原子 | Energy/atom: {energy/len(atoms):.6f} eV/atom")
print(f"最大力 | Max force: {np.max(np.abs(forces)):.6f} eV/Å")
print(f"应力 | Stress: {stress} GPa")

# ========== 详细解析 | Detailed Parsing ==========
class VASPResults:
    """VASP结果解析器 | VASP results parser"""
    
    def __init__(self, outcar_path):
        self.outcar_path = outcar_path
        self.frames = read_vasp_out(outcar_path, index=':')
        
    def get_optimization_history(self):
        """获取优化历史 | Get optimization history"""
        history = []
        for i, atoms in enumerate(self.frames):
            history.append({
                'step': i,
                'energy': atoms.get_potential_energy(),
                'max_force': np.max(np.abs(atoms.get_forces())),
                'cell': atoms.get_cell().tolist(),
                'volume': atoms.get_volume(),
            })
        return history
    
    def get_final_structure(self):
        """获取最终结构 | Get final structure"""
        return self.frames[-1]
    
    def get_magnetization(self):
        """获取磁化信息 (需要ISPIN=2)"""
        # 从OUTCAR手动解析
        magmoms = []
        with open(self.outcar_path, 'r') as f:
            for line in f:
                if 'magnetization (x)' in line:
                    # 解析磁矩数据
                    pass
        return magmoms

# 使用示例
results = VASPResults('OUTCAR')
history = results.get_optimization_history()

# 绘制优化过程
import matplotlib.pyplot as plt

steps = [h['step'] for h in history]
energies = [h['energy'] for h in history]
forces = [h['max_force'] for h in history]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

ax1.plot(steps, energies, 'b-')
ax1.set_ylabel('Energy (eV)')
ax1.set_title('Optimization History')
ax1.grid(True)

ax2.semilogy(steps, forces, 'r-')
ax2.axhline(y=0.01, color='g', linestyle='--', label='Convergence')
ax2.set_xlabel('Step')
ax2.set_ylabel('Max Force (eV/Å)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('optimization_history.png')
```

### 5.2 生成训练数据 | Generate Training Data

```python
"""
从AIMD生成ML势训练数据
Generate ML potential training data from AIMD
"""
import dpdata
from ase.io import read
import numpy as np

# 方法1: 使用dpdata直接读取VASP输出
system = dpdata.LabeledSystem('OUTCAR', fmt='vasp/outcar')
print(f"读取了 {len(system)} 帧数据")

# 分割训练集和验证集
n_frames = len(system)
n_train = int(n_frames * 0.9)
indices = np.random.permutation(n_frames)

train_system = system.sub_system(indices[:n_train])
valid_system = system.sub_system(indices[n_train:])

# 保存为DeepMD格式
train_system.to_deepmd_npy('./training_data')
valid_system.to_deepmd_npy('./validation_data')

# 方法2: 从ASE trajectory转换
frames = read('aimd.traj', index=':')

coords = []
cells = []
energies = []
forces = []

for atoms in frames:
    coords.append(atoms.get_positions())
    cells.append(atoms.get_cell().array)
    energies.append(atoms.get_potential_energy())
    forces.append(atoms.get_forces())

# 创建dpdata系统
system = dpdata.LabeledSystem()
system['atom_names'] = ['Li', 'P', 'S']
system['atom_numbs'] = [12, 4, 16]
system['atom_types'] = np.array([0]*12 + [1]*4 + [2]*16)
system['coords'] = np.array(coords)
system['cells'] = np.array(cells)
system['energies'] = np.array(energies)
system['forces'] = np.array(forces)
system['orig'] = np.zeros(3)

system.to_deepmd_npy('./aimd_training_data')
```

---

## 6. 常见错误与解决 | Common Errors & Solutions

### 6.1 错误诊断 | Error Diagnosis

| 错误 | Error | 可能原因 | Possible Cause | 解决方案 | Solution |
|------|-------|---------|---------------|---------|----------|
| `ZBRENT` | 优化失败 | 初始结构太差 | Poor initial structure | 增大NSW或更换初始结构 |
| `EDDDAV` | 电子步不收敛 | 能带交叉或简并 | Band crossing | 增加SMEMARE或改变ISMEAR |
| `FEXCP` | 电荷密度问题 | 截断能不足 | Insufficient ENCUT | 增加ENCUT |
| `TOO FEW BANDS` | 能带不足 | NBANDS设置太小 | NBANDS too small | 增加NBANDS |
| `PSMAXN` | 赝势问题 | 能量超过赝势范围 | Energy out of range | 检查ENCUT与赝势匹配 |
| `BRIONS` | 矩阵问题 | 优化算法失败 | Optimizer failed | 尝试IBRION=1或IBRION=3 |

### 6.2 调试脚本 | Debugging Script

```python
"""
VASP错误诊断工具 | VASP Error Diagnostic Tool
"""
import re
import os

def diagnose_vasp_error(outcar_path='OUTCAR'):
    """诊断VASP错误 | Diagnose VASP errors"""
    
    if not os.path.exists(outcar_path):
        print("❌ OUTCAR not found!")
        return
    
    with open(outcar_path, 'r') as f:
        content = f.read()
    
    # 检查常见错误
    errors = {
        'ZBRENT': 'Optimization failed - try increasing NSW or improving initial structure',
        'EDDDAV': 'Electronic convergence failed - try increasing NELM or changing ALGO',
        'FEXCP': 'Charge density error - increase ENCUT',
        'TOO FEW BANDS': 'Increase NBANDS in INCAR',
        'PSMAXN': 'Check ENCUT against POTCAR ENMAX',
        'BRIONS': 'Try different IBRION (1, 2, or 3)',
        'sgtf': 'Possible symmetry issue - check ISYM',
    }
    
    found_errors = []
    for error_code, suggestion in errors.items():
        if error_code in content:
            found_errors.append((error_code, suggestion))
    
    if found_errors:
        print("="*60)
        print("发现错误 | Errors Found:")
        print("="*60)
        for code, suggestion in found_errors:
            print(f"\n⚠️  {code}:")
            print(f"   建议 | Suggestion: {suggestion}")
    else:
        print("✓ 未发现常见错误 | No common errors found")
    
    # 检查收敛状态
    if 'reached required accuracy' in content:
        print("\n✓ 计算已成功收敛 | Calculation converged successfully")
    else:
        print("\n⚠️ 计算可能未完成收敛 | Calculation may not be fully converged")

# 运行诊断
diagnose_vasp_error()
```

---

## 7. 练习题 | Exercises

### 练习 1: 收敛性测试 | Exercise 1: Convergence Test

```python
# 为Li2S材料进行完整的收敛性测试
# Complete convergence test for Li2S

from ase.io import read
from ase.build import bulk

# 生成Li2S结构
# Li2S has antifluorite structure
atoms = bulk('Li', 'fcc', a=5.7, cubic=True)
# 需要添加S原子... (使用实际结构文件)

atoms = read('Li2S.cif')

# 任务1: 截断能收敛测试
encut_range = range(300, 701, 50)
# ... (实现测试代码)

# 任务2: k点收敛测试
kpoint_grids = [(2,2,2), (3,3,3), (4,4,4), (5,5,5), (6,6,6)]
# ... (实现测试代码)
```

### 练习 2: 高精度计算 | Exercise 2: High-Precision Calculation

```python
"""
高精度计算设置 | High-Precision Calculation Setup
"""
from ase.calculators.vasp import Vasp

high_precision_calc = Vasp(
    # 极高的精度
    prec='Accurate',
    encut=600,  # 从收敛测试确定
    
    # 严格的收敛标准
    ediff=1e-8,
    ediffg=-0.001,
    
    # k点
    kpts=(8, 8, 8),  # 或更高
    
    # 电子优化
    algo='Normal',  # 更稳定的算法
    nelm=200,
    nelmin=6,
    
    # 离子优化
    ibrion=1,  # 准牛顿法，更精确
    nsw=500,
    
    # 其他
    ismear=-5,  # Tetrahedron with Blöchl corrections
    lreal=.FALSE.,  # 精确的投影
)
```

### 练习 3: 批量计算 | Exercise 3: Batch Calculation

```python
"""
批量DFT计算 | Batch DFT Calculation
"""
import os
from pathlib import Path
from ase.io import read, write
from code_templates.dft_workflow import BatchDFTCalculator, DFTConfig

config = DFTConfig(
    code='vasp',
    functional='PBE',
    encut=520,
    ncores=4
)

# 获取所有结构文件
structure_files = list(Path('./structures').glob('*.vasp'))

# 批量计算
calculator = BatchDFTCalculator(config)
results = calculator.run_batch_relaxation(
    input_files=[str(f) for f in structure_files],
    output_dir='./batch_results'
)

# 分析结果
for result in results:
    if result['status'] == 'success':
        print(f"{result['input_file']}: E = {result['energy']:.4f} eV")
    else:
        print(f"{result['input_file']}: Failed - {result['error']}")
```

---

## 📚 进一步阅读 | Further Reading

- [VASP Wiki](https://www.vasp.at/wiki/index.php/The_VASP_Manual)
- [Quantum ESPRESSO Documentation](https://www.quantum-espresso.org/documentation/)
- [ASE Calculator Documentation](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html)

---

**下一步**: [03 - ML势训练完整指南](03_ml_potential.md)
