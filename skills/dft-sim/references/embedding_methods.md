# 嵌入方法 (Embedding Methods)

## 简介

嵌入方法 (Embedding Methods) 将大体系划分为精确的量子力学(QM)区域和近似的经典力学(MM)区域，实现多尺度、高精度的材料模拟。主要包括QM/MM、ONIOM、DFT嵌入、投影嵌入等方法。

---

## 1. QM/MM 量子力学/分子力学

### 理论基础

QM/MM能量分割:
$$E_{total} = E_{QM}(QM) + E_{MM}(MM) + E_{QM/MM}(QM-MM)$$

**QM-MM相互作用**:
$$E_{QM/MM} = \sum_{i \in QM, j \in MM} \left[ \frac{q_j q_i}{r_{ij}} + 4\epsilon_{ij}\left(\frac{\sigma_{ij}^{12}}{r_{ij}^{12}} - \frac{\sigma_{ij}^{6}}{r_{ij}^{6}}\right) \right]$$

### 边界处理方案

| 方案 | 描述 | 适用场景 |
|-----|------|---------|
| **Mechanical Embedding** | QM计算在真空场中，MM提供位置固定 | 简单界面 |
| **Electronic Embedding** | MM电荷包含在QM哈密顿中 | 极化效应重要 |
| **Polarizable Embedding** | MM区域可极化 | 强相互作用 |

### CP2K实现 (QUICKSTEP)

```bash
# CP2K QM/MM输入示例
&FORCE_EVAL
  METHOD QMMM
  
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME POTENTIAL
    
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &END QS
    
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-7
      MAX_SCF 100
    &END SCF
    
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
  
  &QMMM
    # QM区域定义
    &CELL
      ABC 12.0 12.0 12.0
      PERIODIC XYZ
    &END CELL
    
    &QM_KIND O
      MM_INDEX 1 2 3  # MM原子编号转为QM
    &END QM_KIND
    
    &QM_KIND H
      MM_INDEX 4 5 6
    &END QM_KIND
    
    # QM-MM相互作用
    ECOUPL COULOMB
    USE_GEEP_LIB TRUE
    
    # 边界处理
    &PERIODIC
      &MULTIPOLE
        RCUT 8.0
        EWALD_PRECISION 1.0E-6
      &END MULTIPOLE
    &END PERIODIC
  &END QMMM
  
  &MM
    &FORCEFIELD
      PARMTYPE AMBER
      PARM_FILE_NAME prmtop
      &SPLINE
        RCUT_NB [angstrom] 10.0
      &END SPLINE
    &END FORCEFIELD
    
    &POISSON
      &EWALD
        EWALD_TYPE SPME
        ALPHA 0.35
        GMAX 80
      &END EWALD
    &END POISSON
  &END MM
  
  &SUBSYS
    &CELL
      ABC 40.0 40.0 40.0
      PERIODIC XYZ
    &END CELL
    
    &TOPOLOGY
      COORD_FILE_NAME coord.xyz
      COORD_FILE_FORMAT XYZ
    &END TOPOLOGY
  &END SUBSYS
&END FORCE_EVAL

&MOTION
  &MD
    ENSEMBLE NVT
    STEPS 10000
    TIMESTEP 0.5
    TEMPERATURE 300.0
    
    &THERMOSTAT
      REGION GLOBAL
      &NOSE
        LENGTH 3
        YOSHIDA 3
        TIMECON 100.0
        MTS 2
      &END NOSE
    &END THERMOSTAT
  &END MD
&END MOTION
```

### Amber/TeraChem接口

```bash
#!/bin/bash
# Amber/TeraChem QM/MM流程

# 1. 准备拓扑文件 (LEaP)
cat > leap.in << 'EOF'
source leaprc.protein.ff19SB
source leaprc.water.tip3p

mol = loadpdb protein.pdb
solvatebox mol TIP3PBOX 12.0
saveamberparm mol prmtop inpcrd
quit
EOF
tleap -f leap.in

# 2. QM/MM MD输入
cat > qmmm.in << 'EOF'
&cntrl
  imin=0, irest=0, ntx=1,
  nstlim=10000, dt=0.001,
  ntf=2, ntc=2,
  cut=8.0,
  ntpr=100, ntwx=100, ntwr=1000,
  ntt=1, temp0=300.0, tautp=2.0,
  ntb=1, ntp=0,
  ifqnt=1,          # 开启QM/MM
/
&qmmm
  iqmatoms=1,2,3,4,5,  # QM原子列表
  qmcharge=0,
  qm_theory='EXTERN',   # 外部QM程序
  qmshake=0,
  qm_ewald=1, qm_pme=1,
/
EOF

# 3. TeraChem输入
cat > tc.config << 'EOF'
basis        6-31g*
method       b3lyp
charge       0
spinmult     1
precision    double
scrdir       ./scr
EOF

# 4. 运行
sander -O -i qmmm.in -o qmmm.out -p prmtop -c inpcrd
```

---

## 2. ONIOM 分层计算方法

### 理论框架

$$E_{ONIOM} = E_{high}(QM) + E_{low}(QM+MM) - E_{low}(QM)$$

三层ONIOM:
$$E_{ONIOM3} = E_{high}(small) + E_{med}(medium) + E_{low}(large) - E_{med}(small) - E_{low}(medium)$$

### Gaussian实现

```bash
# ONIOM输入示例 (酶催化)
%chk=enzyme_oniom.chk
%mem=16GB
%nproc=16
#p ONIOM(B3LYP/6-31G(d):AMBER:AMBER) geom=connectivity

Enzyme QM/MM ONIOM calculation

0 1 0 1  (总电荷/自旋 高亮层电荷/自旋)
C    -2.123456   1.234567  -0.567890 H   # 高亮层(H)
N    -1.876543   0.987654   0.123456 H   # 高亮层(H)
O    -0.234567  -1.876543   0.876543 M   # 中间层(M)
H     0.567890  -0.123456   1.234567 L   # 低层(L)
# ... 更多原子

# 连接原子 (link atoms)
H    -3.123456   2.234567  -1.567890 L   # 连接QM-MM

# 几何优化设置
--Link1--
%chk=enzyme_oniom.chk
%mem=16GB
%nproc=16
#p ONIOM(B3LYP/6-31G(d):AMBER:AMBER) opt=modredundant geom=allcheck

# 约束定义
3 F        # 冻结原子3
4 5 6.0 F  # 冻结4-5键长为6.0Å
```

### 柔性约束ONIOM

```bash
# 微迭代优化 (Microiterations)
%chk=protein_oniom.chk
#p ONIOM(B3LYP/6-31G(d):UFF) opt=(micro,maxcycle=100)

# 外层区域MM优化，内层QM优化
# 减少昂贵的QM计算次数
```

---

## 3. DFT嵌入方法

### 3.1 冻结密度嵌入 (FDE)

```bash
# ADF FDE计算示例
$ADFBIN/adf <> eor
Title Water dimer with FDE

Atoms
  O        0.000000    0.000000    0.000000  subfrag=A
  H        0.950000    0.000000    0.000000  subfrag=A
  H       -0.237641    0.918543    0.000000  subfrag=A
  O        0.000000    3.000000    0.000000  subfrag=B
  H        0.950000    3.000000    0.000000  subfrag=B
  H       -0.237641    3.918543    0.000000  subfrag=B
End

Fragments
  A  water_A.t21
  B  water_B.t21
End

FDE
  PW91K          # 动能密度泛函
  FullGrid       # 完整网格积分
  Relaxed        # 松弛嵌入 (相互极化)
End

XC
  GGA PBE
End

eor
```

### 3.2 投影嵌入 (Projection Embedding)

```python
#!/usr/bin/env python3
"""投影嵌入DFT理论概述"""

projection_embedding = """
投影嵌入 (Projection-Based Embedding, Manby, Miller):

核心思想:
1. 整体DFT计算获得总密度 n(r)
2. 定义环境密度 n_env(r) (通常来自 cheaper DFT)
3. 活性区密度 n_act(r) = n(r) - n_env(r)
4. 在活性区进行高精度计算 (CCSD(T), DMC等)

数学形式:
H' = H + Σ_p |φ_p⟩V_emb⟨φ_p|

其中V_emb确保活性区轨道与环境正交:
V_emb = Σ_μ (f_μ - f̃_μ) |χ_μ⟩⟨χ_μ|

软件实现:
- Molpro: 内置嵌入功能
- PySCF: 完整实现 (周期体系)
- CP2K: 密度嵌入
- entos/QCore: 高级嵌入

优势:
- 无边界问题
- 可结合波函数方法
- 适用于强相关体系
"""

# PySCF示例
pyscf_embedding = """
from pyscf import gto, scf, mp, embed

# 总体系
mol = gto.M(atom='''
O  0. 0. 0.
H  0. -0.757 0.587
H  0.  0.757 0.587
F  2.5 0. 0.
''', basis='cc-pvtz')

# DFT计算
mf = scf.RKS(mol)
mf.xc = 'pbe'
mf.kernel()

# 定义活性区 (水分子)
ao_labels = mol.ao_labels()
# 假设前5个原子是水分子
frag_ao_idx = [i for i, s in enumerate(ao_labels) 
               if any(atom in s for atom in ['0 O', '1 H', '2 H'])]

# 投影嵌入
emb = embed.ProjectionEmbedding(mf)
emb.set_fragments(frag_ao_idx, [])
emb.run()

# 在活性区做MP2
mf_act = emb.make_fragment_scf(0)
mp2 = mp.MP2(mf_act)
mp2.kernel()
"""
```

---

## 4. 周期性DFT嵌入

### CP2K量子嵌入

```bash
&FORCE_EVAL
  METHOD QMMM  # 或 EMBED
  
  &DFT
    # 高精度DFT设置 (如hse06, rpa)
    &XC
      &XC_FUNCTIONAL
        &PBE
          SCALE_X 0.0
          SCALE_C 1.0
        &END PBE
        &PBE_HOLE_T_C_LR
          SCALE_X 1.0
          CUTOFF_RADIUS 3.0
        &END PBE_HOLE_T_C_LR
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
  
  # 嵌入式QM/MM
  &QMMM
    ECOUPL COULOMB
    USE_GEEP_LIB TRUE
    
    # 周期性边界
    &PERIODIC
      &MULTIPOLE
        RCUT 10.0
        EWALD_PRECISION 1.0E-8
      &END MULTIPOLE
    &END PERIODIC
  &END QMMM
&END FORCE_EVAL
```

### 激发态嵌入

```python
# QM/MM TDDFT (嵌入环境中的激发态)
tddft_qmmm = """
激发态QM/MM关键考虑:

1. 响应性环境
   - 基态和激发态环境可能不同
   - 需要状态特定的MM参数
   
2. 极化效应
   - 激发态偶极矩通常更大
   - 溶剂/环境响应重要
   
3. 软件选择
   - Q-Chem: 完整QM/MM TDDFT
   - Gaussian: ONIOM-CIS/TDDFT
   - CP2K: 线性响应TDDFT
   - ADF: FDE-TDDFT
"""
```

---

## 5. 实际应用

### 酶催化

```bash
# 酶活性位点QM/MM研究
enzyme_qmmm = """
典型工作流程:

1. 体系准备
   - 晶体结构获取 (PDB)
   - 加氢/质子化状态确定
   - 溶剂化 (TIP3P水盒子)
   - 平衡化MD

2. QM区域选择
   - 底物 + 催化残基
   - 关键辅因子
   - 通常50-200原子

3. 反应路径计算
   - 约束MD/伞形采样
   - 过渡态搜索
   - 自由能计算

4. 验证
   - QM区域大小收敛
   - 不同MM力场比较
   - 实验速率对比
"""
```

### 表面化学

```python
surface_embedding = """
表面QM/MM (VASP+LAMMPS, CP2K):

模型构建:
1. 表层: 高密度DFT (QM)
2. 次表层: 中等精度 (可选)
3. 体相: 力场 (MM)

VASP-COM接口:
- 使用Socket通信
- 实现VASP作为QM引擎
- LAMMPS作为MD驱动

CP2K优势:
- 原生QM/MM支持
- 周期性边界处理完善
- GPU加速可用
"""
```

### 材料缺陷

```python
defect_embedding = """
缺陷嵌入计算:

体系: 大块半导体中的点缺陷

方法1: QM/MM ONIOM
- 缺陷 + 近邻原子 = QM
- 远场 = 力场/弹性

方法2: DFT嵌入
- 超胞计算 = 环境
- 缺陷区域 = 高精度CCSD(T)

方法3: 机器学习加速
- ML势描述环境
- DFT处理缺陷核心

收敛测试:
- QM区域尺寸 (~5-10Å半径)
- MM力场选择
- 长程静电处理
"""
```

---

## 6. 分析工具

### 能量分解

```python
#!/usr/bin/env python3
"""QM/MM能量分解分析"""

def analyze_qmmm_energy(logfile):
    """解析QM/MM能量组成"""
    
    energy_components = {
        'E_QM': 0.0,      # QM区域内部
        'E_MM': 0.0,      # MM区域内部
        'E_QMMM_elec': 0.0,  # QM-MM静电
        'E_QMMM_vdw': 0.0,   # QM-MM范德华
        'E_bonded': 0.0,     # 连接原子修正
    }
    
    with open(logfile) as f:
        for line in f:
            if 'QM energy' in line:
                energy_components['E_QM'] = float(line.split()[-1])
            elif 'MM energy' in line:
                energy_components['E_MM'] = float(line.split()[-1])
            elif 'QMMM electrostatic' in line:
                energy_components['E_QMMM_elec'] = float(line.split()[-1])
            elif 'QMMM van der Waals' in line:
                energy_components['E_QMMM_vdw'] = float(line.split()[-1])
    
    E_total = sum(energy_components.values())
    
    print("="*60)
    print("QM/MM Energy Decomposition")
    print("="*60)
    for key, val in energy_components.items():
        print(f"{key:<20}: {val:12.4f} kcal/mol")
    print("-"*60)
    print(f"{'TOTAL':<20}: {E_total:12.4f} kcal/mol")
    
    return energy_components
```

### 轨迹分析

```python
def analyze_qmmm_trajectory(traj_file, top_file):
    """分析QM/MM MD轨迹"""
    import MDAnalysis as mda
    
    u = mda.Universe(top_file, traj_file)
    
    # 选择QM区域
    qm_atoms = u.select_atoms('resname SUB')
    
    # 分析RMSD
    from MDAnalysis.analysis import rms
    R = rms.RMSD(qm_atoms, qm_atoms, select='backbone')
    R.run()
    
    # QM-MM相互作用能随时间变化
    energies = []
    for ts in u.trajectory:
        # 提取各组分能量
        # 需要QM/MM软件特定输出
        pass
    
    return R.rmsd, energies
```

---

## 7. 最佳实践

### QM区域选择原则

```python
qm_selection_guide = """
QM区域选择标准:

1. 化学反应性
   - 必须包含所有参与键断裂/形成的原子
   - 包含直接配位的原子

2. 极化效应
   - 考虑电荷转移程度
   - 可能需要扩展至第二配位层

3. 尺寸收敛
   - 测试QM区域大小 (~100-200原子)
   - 能量/结构性质收敛

4. 边界位置
   - 避免切断共价键 (或使用连接原子)
   - 优先选择非极性键 (C-C, C-H)

5. 对称性
   - 利用对称性减少计算量
   - 注意周期性边界
"""
```

### 常见陷阱

| 问题 | 原因 | 解决 |
|-----|------|------|
| QM-MM电荷泄漏 | 电子密度溢出到MM区 | 增大QM区或使用约束 |
| 边界人工效应 | 连接原子或边界设置不当 | 测试不同边界方案 |
| 极化不足 | Mechanical embedding忽略MM极化 | 改用electronic embedding |
| SCF不收敛 | QM-MM电荷相互作用 | 调整初始猜测或混合 |
| 能量漂移 | MM力场参数不当 | 验证力场参数 |

### 推荐软件组合

| 应用场景 | 推荐方案 |
|---------|---------|
| 酶催化 | Amber/TeraChem, CP2K |
| 表面化学 | CP2K, VASP+LAMMPS |
| 溶液化学 | Gromacs/CP2K, Q-Chem |
| 激发态 | Q-Chem, Gaussian, ADF |
| 周期性体系 | CP2K, Quantum Espresso+MM |

---

## 参考资源

- CP2K QM/MM: https://manual.cp2k.org/
- Amber QM/MM: https://ambermd.org/tutorials/advanced/tutorial20/
- Gaussian ONIOM: https://gaussian.com/oniom/
- ADF FDE: https://www.scm.com/doc/ADF/Input/Fragm_Embed.html
- Review: Senn, Thiel, QM/MM Methods for Biomolecular Systems (Angew. Chem.)
- Review: Libisch, Huang, Carter, Embedded Correlated Wavefunction Methods (Chem. Rev.)

---

*文档版本: 1.0*
*最后更新: 2026-03-08*
