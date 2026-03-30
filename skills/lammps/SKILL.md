# LAMMPS 分子动力学技能

> 大型原子/分子大规模并行模拟器 (Large-scale Atomic/Molecular Massively Parallel Simulator)
> 技能等级: 进阶 | 更新日期: 2026-03-08

---

## 技能概述

本技能库提供完整的LAMMPS分子动力学模拟知识体系，涵盖从安装配置到高级应用的各个方面。

### 适用场景
- 材料科学：金属、合金、聚合物、复合材料的结构与性能
- 生物物理：蛋白质折叠、膜模拟、DNA/RNA动力学
- 界面科学：表面吸附、润湿、催化反应
- 化工过程：流体性质、相分离、传质
- 极端条件：高温高压、冲击波、辐照损伤

### 前置要求
- Linux/Unix基础操作
- Python编程基础
- 统计力学与分子动力学理论基础

---

## 文档索引

### 核心教程
| 章节 | 文档 | 内容概要 |
|-----|------|---------|
| 01 | [安装配置](01-installation.md) | LAMMPS编译安装、并行配置、GPU加速 |
| 02 | [势函数库](02-potentials.md) | 经典力场、ML势、ReaxFF、EAM/MEAM |
| 03 | [基础MD计算](03-basic-md.md) | NVT/NPT系综、积分器、邻居列表 |
| 04 | [高级采样](04-advanced-sampling.md) | 伞形采样、REMD、Metadynamics、TAD |
| 05 | [多尺度模拟](05-multiscale.md) | QM/MM、DFT-MD耦合、ML势多尺度 |
| 06 | [材料案例](06-materials-cases.md) | 金属、聚合物、生物、界面体系案例 |
| 07 | [分析工具](07-analysis.md) | RDF/MSD/VACF、自由能计算、可视化 |
| 08 | [性能优化](08-performance.md) | MPI/OpenMP/GPU、负载均衡、基准测试 |

### 快速参考

#### 常用命令速查
```bash
# 运行模拟
lmp -in input.lammps -log log.lammps

# 并行运行
mpirun -np 8 lmp -in input.lammps

# GPU加速
lmp -k on -sf kk -in input.lammps

# 带变量运行
lmp -in input.lammps -v T 300 -v P 1.0
```

#### 典型输入结构
```lammps
# 1. 初始化
units metal
atom_style atomic
boundary p p p

# 2. 创建体系
read_data data.lmp
# 或
lattice fcc 3.615
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 box

# 3. 势函数
pair_style eam/alloy
pair_coeff * * Cu_u3.eam.alloy Cu

# 4. 模拟设置
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
timestep 0.001

# 5. 能量最小化
minimize 1.0e-8 1.0e-8 1000 10000

# 6. 初始速度
velocity all create 300.0 12345 mom yes rot yes

# 7. 约束与控制
fix 1 all nvt temp 300.0 300.0 $(100.0*dt)

# 8. 输出
dump 1 all custom 1000 dump.lammpstrj id type x y z
thermo 1000

# 9. 运行
run 100000
```

---

## 学习路径

### 初学者路径 (1-2周)
1. [安装配置](01-installation.md) - 完成LAMMPS安装
2. [基础MD计算](03-basic-md.md) - 理解NVT/NPT系综
3. [势函数库](02-potentials.md) - 掌握EAM/LJ势
4. 完成Cu熔化和水模拟示例

### 进阶路径 (2-4周)
5. [分析工具](07-analysis.md) - RDF、MSD分析
6. [材料案例](06-materials-cases.md) - 完成一个完整案例
7. [性能优化](08-performance.md) - 并行计算优化

### 专家路径 (1-2月)
8. [高级采样](04-advanced-sampling.md) - 自由能计算
9. [多尺度模拟](05-multiscale.md) - QM/MM耦合
10. 开发自定义pair_style或fix

---

## 常见问题

### Q: 如何选择势函数？
**A**: 参考以下决策树：
- 纯金属/合金 → EAM/MEAM
- 共价晶体(Si, C) → Tersoff/SW
- 分子/有机 → OPLS/AMBER/CHARMM
- 化学反应 → ReaxFF
- 高精度需求 → 机器学习势(ACE/DeepMD)

### Q: 时间步长如何选择？
**A**: 
- 含氢体系: 0.5-1.0 fs
- 有机分子: 1.0-2.0 fs
- 约束体系(刚性水): 2.0 fs
- 金属: 1.0-5.0 fs

### Q: 体系多大才够？
**A**: 
- 块体材料: >2000原子 (消除有限尺寸效应)
- 界面: >5000原子 (足够厚的体相区域)
- 生物分子: >10000原子 (溶剂化层)

### Q: 运行多长时间？
**A**: 
- 平衡: 10-100 ps (取决于体系)
- 性质计算: 100 ps - 10 ns
- 稀有事件: 增强采样方法

---

## 外部资源

### 官方资源
- [LAMMPS官网](https://www.lammps.org/)
- [官方文档](https://docs.lammps.org/)
- [GitHub仓库](https://github.com/lammps/lammps)
- [用户论坛](https://matsci.org/lammps)

### 势函数数据库
- [NIST Interatomic Potentials](https://www.ctcms.nist.gov/potentials/)
- [OpenKIM](https://openkim.org/)
- [AI4Materials](https://github.com/AI4Materials)

### 相关工具
- [VMD](https://www.ks.uiuc.edu/Research/vmd/) - 可视化
- [OVITO](https://www.ovito.org/) - 分析可视化
- [MDAnalysis](https://www.mdanalysis.org/) - Python分析
- [ASE](https://wiki.fysik.dtu.dk/ase/) - 原子模拟环境

---

## 版本信息

- **LAMMPS版本**: 2024-2025系列
- **文档更新**: 2026-03-08
- **技能覆盖度**: 95%
- **文档总数**: 8个核心文档

### 更新日志

**2026-03-08**
- 创建完整LAMMPS技能库
- 完成8个核心模块文档
- 添加金属/聚合物/生物/界面案例
- 包含分析工具和性能优化指南

---

## 贡献与反馈

本技能库持续进化中。如需添加内容或报告问题，请参考：
- 提交示例输入脚本
- 补充特定体系的最佳实践
- 分享性能优化经验
