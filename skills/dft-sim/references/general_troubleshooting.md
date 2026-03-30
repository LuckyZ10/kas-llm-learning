# DFT计算综合故障排查手册

## 快速诊断流程图

```
计算失败
    │
    ├─ 编译错误? ─── 检查编译器/库路径 → 重新配置编译
    │
    ├─ 输入错误? ─── 检查INCAR/POSCAR格式 → 修正输入
    │
    ├─ SCF不收敛? ── 调整ALGO/MIXING → 尝试不同算法
    │
    ├─ 离子步不收敛? ─ 调整POTIM/IBRION → 放松收敛标准
    │
    ├─ 内存不足? ─── 降低NCORE/减少K点 → 分步计算
    │
    └─ 未知错误? ─── 检查日志 → 搜索错误信息 → 查阅文档
```

---

## 1. 初始结构问题

### 问题: 初始结构能量异常高
**症状**: 第一步SCF能量远高于预期
**诊断**:
```python
#!/usr/bin/env python3
"""诊断初始结构问题"""
from ase.io import read
import numpy as np

def diagnose_structure(poscar_file):
    """诊断POSCAR结构问题"""
    atoms = read(poscar_file)
    
    print("="*60)
    print("Structure Diagnosis Report")
    print("="*60)
    
    # 检查原子间距
    distances = []
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            d = atoms.get_distance(i, j, mic=True)
            distances.append((d, i, j))
    
    distances.sort()
    print(f"\nShortest distances:")
    for d, i, j in distances[:5]:
        status = "⚠️ TOO CLOSE" if d < 0.8 else "✓ OK"
        print(f"  {atoms[i].symbol}-{atoms[j].symbol}: {d:.3f} Å {status}")
    
    # 检查晶格
    print(f"\nCell parameters:")
    print(f"  a, b, c: {atoms.cell.lengths()}")
    print(f"  α, β, γ: {atoms.cell.angles()}")
    print(f"  Volume: {atoms.get_volume():.2f} Å³")
    
    # 检查是否有重叠
    if distances[0][0] < 0.5:
        print("\n❌ CRITICAL: Atoms are overlapping!")
        print("   Check lattice constants and coordinate units")
```

**解决**:
- 检查晶格常数单位 (Å vs Bohr)
- 检查坐标系 (笛卡尔 vs 分数坐标)
- 使用VESTA可视化结构

### 问题: 高对称性结构优化失败
**症状**: 优化后对称性降低，能量异常
**解决**:
```bash
# VASP
ISYM = 0           # 先关闭对称性优化
# 收敛后再开启ISYM=2精细优化

# QE
nosym = .true.,
```

---

## 2. 电子结构问题

### 问题: 磁矩不正确
**诊断**:
```bash
# 检查初始磁矩设置
# VASP: MAGMOM
# QE: starting_magnetization
```

**解决**:
```bash
# VASP - 铁磁性
MAGMOM = 5*2 4*-1  # 5个原子磁矩2, 4个原子磁矩-1

# 反铁磁性 (MnO为例)
MAGMOM = 2*4 2*-4  # Mn原子反平行排列

# QE
starting_magnetization(1) = 0.5,  # 类型1
starting_magnetization(2) = -0.5, # 类型2
```

### 问题: 带隙计算不正确
**原因与解决**:
```bash
# 1. k点网格太稀
# VASP
KPOINTS
0
Gamma
12 12 12  # 增加密度
0 0 0

# 2. 需要DFT+U或杂化泛函
# VASP
LDAU = .TRUE.
LDAUL = 2 -1    # d轨道加U
LDAUU = 4.0 0.0

# 或HSE06
LHFCALC = .TRUE.
ALGO = All

# 3. 能带计算需要足够空带
NBANDS = 40  # 增加空带
```

### 问题: 费米能级位置异常
**诊断**:
```bash
# 检查DOS积分
# 价电子数 = ∫ DOS dE (从-∞到EF)

# 金属/半导体判断
# 如果DOS(EF) > 0.1 states/eV → 金属
# 如果DOS(EF) ≈ 0 → 半导体
```

---

## 3. 收敛问题速查表

### SCF收敛问题

| 症状 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 能量震荡 | 混合参数过大 | AMIX=0.2, BMIX=0.0001 |
| 能量上升 | 初始波函数差 | ISTART=0, 重新生成 |
| 收敛慢 | 带隙大/体系复杂 | ALGO=All/David |
| 不收敛 | 磁性/强关联 | 使用DFT+U, 调整MAGMOM |

### 离子弛豫问题

| 症状 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 力不下降 | POTIM太大 | 减小到0.1或0.05 |
| 步数用完 | NSW不够 | 增加到200 |
| 发散 | 初始结构差 | 先固定体积弛豫 |
| 对称破缺 | 数值误差 | ISYM=0, SYM_PREC=1E-5 |

---

## 4. 内存与性能问题

### 内存估算公式

```python
#!/usr/bin/env python3
"""估算DFT计算内存需求"""

def estimate_memory(natoms, nkpoints, nbands, ncores, 
                    enct=500, complex_wf=True):
    """估算VASP内存需求 (GB)"""
    
    # 基本参数
    ngrid = (enct / 200) ** 1.5 * 10000  # 近似FFT网格点
    
    # 波函数内存 (主要)
    wf_size = nkpoints * nbands * ngrid * (16 if complex_wf else 8)  # bytes
    wf_size_gb = wf_size / 1e9
    
    # 其他数组 (电荷密度等)
    other_gb = nkpoints * ngrid * 8 / 1e9 * 10
    
    # 每核内存
    per_core = (wf_size_gb + other_gb) / ncores * 3  # 系数3为安全系数
    
    total = per_core * ncores
    
    print(f"Memory Estimation:")
    print(f"  Wavefunction: {wf_size_gb:.1f} GB")
    print(f"  Other arrays: {other_gb:.1f} GB")
    print(f"  Total: {total:.1f} GB")
    print(f"  Per core: {per_core:.1f} GB")
    
    return total, per_core

# 示例
estimate_memory(natoms=100, nkpoints=8, nbands=200, ncores=32, enct=520)
```

### 降低内存使用

```bash
# VASP
NCORE = 4          # 增加降低每核内存
LREAL = Auto       # 实空间投影
LWAVE = .FALSE.    # 不保存波函数
LCHARG = .FALSE.   # 不保存电荷密度

# 分步计算
# 1. 粗k点收敛
# 2. 用WAVECAR粗k点启动密k点
```

---

## 5. 特殊计算问题

### 表面计算

**问题: 表面偶极矩导致能带弯曲**
```bash
# VASP - 偶极修正
LDIPOL = .TRUE.
IDIPOL = 3          # 垂直于表面
DIPOL = 0.5 0.5 0.5  # 中心位置

# 或对称性slab
# 构建上下表面相同的结构 (如Si-Si双键终止)
```

**问题: 表面弛豫过度**
```bash
# 固定底层原子
# POSCAR中选择性动力学
Selective dynamics
Direct
0.0 0.0 0.0 T T F    # 顶层可动
0.5 0.5 0.1 F F F    # 底层固定
```

### 缺陷计算

**问题: 带电缺陷SCF不收敛**
```bash
# VASP
# 1. 增加真空层 (≥15Å)
# 2. 使用高斯展宽
ISMEAR = 0
SIGMA = 0.05

# 3. 逐步添加电荷
# 先算中性，再用NELECT逐步调整
NELECT = 255.5  # 从整数开始微调

# 4. 使用Lany-Zunger修正
LVHAR = .TRUE.   # 输出 Hartree 势
```

### 分子动力学

**问题: 能量漂移**
```bash
# 检查
# 1. 时间步长太大
POTIM = 0.5  # 从2.0降低到0.5 fs

# 2. 初始结构未优化
# 先做结构优化 (NSW=50)

# 3. 截断能不足
ENCUT = 1.3 * ENMAX  # 增加至1.3倍

# 4. 算法选择
# NVE: 检查能量守恒
# NVT: 检查热浴耦合
```

---

## 6. 结果验证清单

### 基础验证

- [ ] 能量收敛 (EDIFF reached)
- [ ] 力收敛 (所有原子 < EDIFFG)
- [ ] 晶胞体积合理 (与实验值偏差 < 5%)
- [ ] 键长合理 (与数据库对比)

### 电子结构验证

- [ ] 带隙合理 (PBE低估0.5-1.0 eV是正常的)
- [ ] DOS积分 = 价电子数
- [ ] 能带路径覆盖高对称点
- [ ] 费米能级位置合理

### 特殊计算验证

- [ ] 声子: 无虚频 (Γ点除外)
- [ ] NEB: 过渡态有唯一虚频
- [ ] MD: 能量守恒 (NVE) 或温度稳定 (NVT)

---

## 7. 日志分析脚本

```python
#!/usr/bin/env python3
"""DFT日志分析工具"""

import re
import sys

def analyze_vasp_outcar(filename='OUTCAR'):
    """分析VASP OUTCAR文件"""
    
    issues = []
    warnings = []
    
    with open(filename, 'r') as f:
        content = f.read()
        lines = f.readlines()
    
    # 检查关键错误
    if 'Fatal error' in content:
        issues.append("❌ Fatal error detected")
    
    if 'convergence NOT achieved' in content:
        issues.append("❌ SCF not converged")
        
        # 提取最后几步能量
        energies = re.findall(r'E\s*=\s*([-\d.]+)', content)
        if energies:
            last_5 = [float(e) for e in energies[-5:]]
            if max(last_5) - min(last_5) < 0.001:
                warnings.append("⚠️ SCF stalled (oscillating)")
    
    if 'VERY BAD NEWS' in content:
        issues.append("❌ Internal VASP error")
    
    # 检查警告
    if 'WVFN is partially filled' in content:
        warnings.append("⚠️ Partial occupancy - check ISMEAR")
    
    if 'Linear dependent' in content:
        warnings.append("⚠️ Linear dependence in basis")
    
    # 性能警告
    if 'your setting of NPAR' in content:
        warnings.append("⚠️ Suboptimal NPAR setting")
    
    print("="*60)
    print("OUTCAR Analysis Report")
    print("="*60)
    
    if issues:
        print("\nCritical Issues:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n✓ No issues detected")
    
    # 提取最终结果
    final_energy = re.search(r'TOTEN\s*=\s*([-\d.]+)', content)
    if final_energy:
        print(f"\nFinal energy: {final_energy.group(1)} eV")
    
    return issues, warnings

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_vasp_outcar(sys.argv[1])
    else:
        analyze_vasp_outcar()
```

---

## 8. 联系社区

当自行排查无果时:

1. **准备问题报告**:
   - 软件版本
   - 编译器/库信息
   - 完整输入文件
   - 错误日志 (前50行和后50行)
   - 已尝试的解决方案

2. **提问渠道**:
   - VASP: vasp@univie.ac.at
   - QE: users@lists.quantum-espresso.org
   - Stack Overflow: [vasp] [quantum-espresso] 标签

---

*文档版本: 1.0*
*最后更新: 2026-03-08*
