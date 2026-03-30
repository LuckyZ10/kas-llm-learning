# 多尺度耦合方法 (QM/MM 与 ONIOM)

本文档介绍多尺度计算方法，包括QM/MM和ONIOM，用于处理大体系或复杂环境中的化学问题。

---

## 1. 多尺度方法概述

### 1.1 为什么需要多尺度

**纯DFT的局限**:
- 计算成本 ~ O(N³)，体系大小受限 (~1000原子)
- 难以描述长程相互作用 (溶剂、蛋白质环境)
- 界面/表面问题需要大体系模型

**多尺度优势**:
| 方法 | QM区域 | MM区域 | 总体系 | 适用场景 |
|------|--------|--------|--------|----------|
| 纯DFT | 全部 | - | ~500原子 | 团簇、单胞 |
| QM/MM | ~100原子 | ~10000原子 | ~10000原子 | 溶液、酶反应 |
| ONIOM | ~50原子 | ~500原子 | ~500原子 | 局域性质 |
| 机器学习势 | - | - | ~100万原子 | 大尺度MD |

### 1.2 QM/MM基本原理

**能量分区**:
$$E_{total} = E_{QM} + E_{MM} + E_{QM/MM}$$

其中:
- $E_{QM}$: 量子力学区域能量 (DFT/半经验)
- $E_{MM}$: 分子力学区域能量 (力场)
- $E_{QM/MM}$: QM-MM相互作用

**QM/MM相互作用**:
$$E_{QM/MM} = \sum_{i \in QM, j \in MM} \left[ \frac{q_j}{|r_i - R_j|} + V_{LJ}(r_{ij}) \right]$$

---

## 2. 静电嵌入QM/MM

### 2.1 力学嵌入 vs 静电嵌入

| 类型 | QM区域感受MM | 实现难度 | 精度 |
|------|--------------|----------|------|
| 力学嵌入 | 无 | 低 | 低 |
| 静电嵌入 | MM电荷 | 中 | 高 |
| 极化嵌入 | MM极化 | 高 | 最高 |

### 2.2 CP2K中的QM/MM

**输入文件示例**:
```
&FORCE_EVAL
  METHOD QMMM
  
  &DFT
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &END QS
    
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 100
    &END SCF
    
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
  
  &MM
    &FORCEFIELD
      PARMTYPE AMBER
      PARM_FILE_NAME prmtop
      &SPLINE
        RCUT_NB 10.0
      &END SPLINE
    &END FORCEFIELD
    
    &POISSON
      &EWALD
        EWALD_TYPE SPME
        GMAX 80
      &END EWALD
    &END POISSON
  &END MM
  
  &QMMM
    &CELL
      ABC 20.0 20.0 20.0
    &END CELL
    
    &QM_KIND O
      MM_INDEX 1 2 3 4 5
    &END QM_KIND
    
    &QM_KIND H
      MM_INDEX 6 7 8 9 10
    &END QM_KIND
    
    ECOUPL COULOMB          ! 静电嵌入
    &PERIODIC
      GMAX 80
    &END PERIODIC
  &END QMMM
  
  &SUBSYS
    &TOPOLOGY
      COORD_FILE_NAME coord.xyz
    &END TOPOLOGY
  &END SUBSYS
&END FORCE_EVAL
```

### 2.3 CP2K运行脚本

```bash
#!/bin/bash
# qmmm_cp2k.sh

module load cp2k

export OMP_NUM_THREADS=4

# 运行QM/MM计算
mpirun -np 16 cp2k.popt -i qmmm.inp -o qmmm.out

# 分析结果
grep "Total FORCE_EVAL" qmmm.out | tail -1
```

### 2.4 DFTB+中的QM/MM

**输入文件**:
```
# dftb_in.hsd
Geometry = GenFormat {
  <<< "geometry.gen"
}

Driver = {}

Hamiltonian = DFTB {
  SCC = Yes
  SCCTolerance = 1e-6
  MaxAngularMomentum = {
    O = "p"
    H = "s"
    C = "p"
    N = "p"
  }
  SlaterKosterFiles = Type2FileNames {
    Prefix = "/path/to/slako/"
    Separator = "-"
    Suffix = ".skf"
  }
}

# QM/MM设置
QM/MM = {
  # QM原子索引
  QMAtoms = {1:10}
  
  # MM力场文件
  MMPotentialFile = "mm_potential.dat"
  
  # 静电嵌入
  Electrostatic = Yes
}
```

---

## 3. ONIOM方法

### 3.1 ONIOM原理

**能量外推公式**:
$$E_{ONIOM} = E_{real,low} + E_{model,high} - E_{model,low}$$

其中:
- $E_{real,low}$: 大体系低精度计算
- $E_{model,high}$: 小体系高精度计算
- $E_{model,low}$: 小体系低精度计算

**典型组合**:
| 应用 | High Level | Low Level |
|------|------------|-----------|
| 有机反应 | DFT/B3LYP | 半经验/力场 |
| 金属配合物 | CCSD(T) | DFT |
| 酶催化 | DFT+MM | MM |

### 3.2 Gaussian中的ONIOM

**输入文件示例**:
```
%chk=oniom_calc.chk
%mem=32GB
%nprocshared=16
#p ONIOM(B3LYP/6-31G(d):AMBER)=EmbedCharge Opt

ONIOM optimization of enzyme active site

0 1 0 1               ! 总电荷 自旋  QM电荷 QM自旋
C-C_3                0    2.345    1.234    0.567 H   ! H=High level
O-O_3                0    1.987    2.456    1.234 H
N-N_3                0    3.456    0.987    2.345 H
C-C_3                0    4.567    1.876    0.432 L   ! L=Low level
H-H_                 0    5.123    2.345    1.876 L
...

# 连接原子 (如果需要)
H-H_                 0    2.500    1.500    0.800 L   ! Link atom
```

**分层策略**:
```
# 三层ONIOM
#p ONIOM(CCSD(T)/cc-pVTZ:B3LYP/6-31G(d):AMBER)=EmbedCharge

Atom1  H   ! High: CCSD(T)
Atom2  H
Atom3  M   ! Medium: DFT
Atom4  M
Atom5  L   ! Low: MM
```

### 3.3 ORCA中的ONIOM

**输入文件**:
```
! ONIOM B3LYP D3 def2-TZVP TightSCF

%geom
  inhess unit
  inhessname "model.hess"
end

%oniom
  Model_Typ QM                           ! 模型层: QM
  HighLevel_Method "B3LYP D3 def2-TZVP"  ! 高精度方法
  LowLevel_Method "BP86 D3 def2-SVP"     ! 低精度方法
  
  Charge_Mult_High 0 1                   ! 模型层电荷和自旋
  Charge_Mult_Low 0 1                    ! 实层电荷和自旋
  
  # 分区定义
  SubRegion 1                            ! QM区域
    Atoms {0:10}
  End
  
  SubRegion 2                            ! 连接区域
    Atoms {11:15}
  End
end

* xyz 0 1
  C    0.000    0.000    0.000
  O    1.200    0.000    0.000
  ...
*
```

---

## 4. 特殊边界处理

### 4.1 连接原子 (Link Atoms)

**问题**: QM/MM边界切断共价键

**解决方案**:
1. **连接原子**: 在断键处添加H原子
2. **冻结连接原子**: 优化时不移动
3. **校正能量**: 计算链接原子修正

**CP2K中的实现**:
```
&QMMM
  &LINK
    QM_INDEX  10
    MM_INDEX  11
    LINK_TYPE IMOMM          ! 积分MO/MM方法
    ALPHA_IMOMM 1.5
  &END LINK
&END QMMM
```

### 4.2 边界区域平滑

**自适应分区 (Adaptive Partitioning)**:
```
&QMMM
  &ADAPTIVE
    ADAPTIVE_METHOD DIFFUSION
    RADIUS 6.0                 ! 自适应区域半径
    ADAPTIVE_SMOOTH 2.0        ! 平滑过渡宽度
  &END ADAPTIVE
&END QMMM
```

---

## 5. 应用案例

### 5.1 酶催化反应

**体系**: 碳酸酐酶中的CO₂水合反应
- **QM区域**: 活性位点Zn²⁺ + 配体 (~50原子)
- **MM区域**: 蛋白质+溶剂 (~10000原子)
- **方法**: DFT(B3LYP)/MM(AMBER)

**CP2K输入关键部分**:
```
&QMMM
  &QM_KIND Zn
    MM_INDEX 1000
  &END QM_KIND
  
  &QM_KIND O
    MM_INDEX 1001 1002 1003  # 配位水分子
  &END QM_KIND
  
  &QM_KIND N
    MM_INDEX 500 501 502     # 组氨酸残基
  &END QM_KIND
  
  ECOUPL COULOMB
  USE_GEEP_LIB YES           ! 高斯展开静电势
&END QMMM
```

### 5.2 溶剂化效应

**显式溶剂QM/MM**:
```
# 第一层水分子: QM (DFT)
# 外层水分子: MM (TIP3P)

&QMMM
  &QM_KIND O
    MM_INDEX 1:32           ! 第一溶剂层
  &END QM_KIND
  
  &QM_KIND H
    MM_INDEX 33:96
  &END QM_KIND
  
  # 周期性边界条件
  &PERIODIC
    &POISSON
      POISSON_SOLVER PERIODIC
    &END POISSON
  &END PERIODIC
&END QMMM
```

### 5.3 表面吸附

**表面催化QM/MM**:
```
# QM: 吸附物 + 表层金属
# MM: 体相金属 (EAM势)

&MM
  &FORCEFIELD
    &EAM
      ATOM_TYPE Pt
      POT_FILE_NAME Pt_u3.eam
    &END EAM
  &END FORCEFIELD
&END MM

&QMMM
  # QM: 表层3层Pt + CO吸附物
  &QM_KIND Pt
    MM_INDEX 100:200
  &END QM_KIND
  
  &QM_KIND C
    MM_INDEX 201
  &END QM_KIND
  
  &QM_KIND O
    MM_INDEX 202
  &END QM_KIND
&END QMMM
```

---

## 6. 验证与基准测试

### 6.1 精度验证

**测试方法**:
1. **能量一致性**: QM/MM vs 纯QM (小体系)
2. **几何结构**: 键长/键角对比
3. **反应能垒**: 与实验或高阶方法对比

**Python验证脚本**:
```python
def validate_qmmm(qmmm_energy, full_qm_energy, mm_energy):
    """
    验证QM/MM能量一致性
    """
    error = abs(qmmm_energy - full_qm_energy)
    relative_error = error / abs(full_qm_energy) * 100
    
    print(f"QM/MM能量: {qmmm_energy:.6f} Ha")
    print(f"纯QM能量:  {full_qm_energy:.6f} Ha")
    print(f"绝对误差:  {error:.6f} Ha")
    print(f"相对误差:  {relative_error:.4f}%")
    
    if relative_error < 1.0:
        print("✓ 精度合格")
    else:
        print("✗ 精度不足，检查边界设置")
    
    return relative_error
```

### 6.2 基准测试结果

| 体系 | 纯QM (DFT) | QM/MM | 误差 | 加速比 |
|------|------------|-------|------|--------|
| 水二聚体 | -152.345 | -152.341 | 0.003% | 1x |
| (H₂O)₂₀ | -3046.12 | -3045.89 | 0.008% | 10x |
| 蛋白质活性位点 | N/A | - | - | 100x |

---

## 7. 常见软件对比

| 软件 | QM方法 | MM力场 | 特点 | 适用场景 |
|------|--------|--------|------|----------|
| CP2K | DFT/GPW | AMBER/GROMOS | 高效，周期性 | 材料、溶液 |
| Gaussian | 全方法 | AMBER/OPLS | 高精度QM | 有机化学 |
| ORCA | 全方法 | 多种 | 开源免费 | 过渡金属 |
| Q-Chem | 全方法 | CHARMM | 先进算法 | 光化学 |
| NWChem | 全方法 | 多种 | 并行优秀 | 大体系 |
| DFTB+ | DFTB | 多种 | 快速 | 大体系筛选 |
| Amber | 半经验 | AMBER | 生物模拟 | 蛋白质MD |

---

## 8. 最佳实践

### 8.1 QM区域选择原则

1. **包含所有参与化学键变化的原子**
2. **包含电荷转移相关的原子**
3. **边界避免切断极性键**
4. **最小化连接原子数量**

### 8.2 常见陷阱

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| 边界电荷泄漏 | 能量不收敛 | 使用Gaussian模糊电荷 |
| 连接原子漂移 | 几何优化失败 | 冻结或约束连接原子 |
| QM/MM不匹配 | 界面处力异常 | 增加过渡区域 |
| 周期性边界错误 | 静电能不收敛 | 使用Ewald求和 |

---

## 参考

1. Senn, Thiel, *Angew. Chem.* 48, 1198 (2009) - QM/MM综述
2. Lin, Truhlar, *Theor. Chem. Acc.* 117, 185 (2007) - QM/MM方法
3. CP2K Manual: [QM/MM](https://manual.cp2k.org/#/QM/MM)
4. Gaussian User's Guide: [ONIOM](https://gaussian.com/oniom/)
