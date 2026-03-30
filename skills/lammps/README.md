# LAMMPS 分子动力学技能库

> 大型原子/分子大规模并行模拟器 (Large-scale Atomic/Molecular Massively Parallel Simulator)
> 版本: 2024-2025 系列 | 更新: 2026-03-08
> 适用平台: Linux/macOS/HPC集群

---

## 📚 文档目录

| 章节 | 文档 | 内容 |
|-----|------|------|
| 01 | [安装配置](01-installation.md) | LAMMPS编译安装、并行配置、GPU加速 |
| 02 | [势函数库](02-potentials.md) | 经典力场、ML势(ACE/DeepMD)、ReaxFF、EAM/MEAM |
| 03 | [基础MD计算](03-basic-md.md) | NVT/NPT系综、积分器、邻居列表、输出控制 |
| 04 | [高级采样](04-advanced-sampling.md) | 伞形采样、REMD、Metadynamics、TAD |
| 05 | [多尺度模拟](05-multiscale.md) | QM/MM耦合、DFT-MD、机器学习势多尺度 |
| 06 | [材料案例](06-materials-cases.md) | 金属、聚合物、生物分子、界面体系 |
| 07 | [ML势训练](07-ml-potential-training.md) | DeepMD/DP-GEN主动学习工作流 |
| 08 | [大规模并行](08-large-scale-parallel.md) | MPI/OpenMP/GPU混合优化、百亿原子 |
| 09 | [材料数据库](09-materials-databases.md) | OpenKIM/Matlantis/AFLOW集成 |
| 10 | [可视化后处理](10-visualization-postprocessing.md) | OVITO/VMD/Pymatgen分析 |
| 11 | [特定应用案例](11-specific-applications.md) | 电池/催化/高熵合金深度案例 |
| - | [示例脚本](examples.md) | 完整输入文件示例 |
| - | [快速索引](INDEX.md) | 命令速查、主题索引 |
| - | [技能主文档](SKILL.md) | 学习路径、常见问题 |

---

## 🚀 快速开始

```bash
# 验证安装
lmp -h

# 基础运行
lmp -in input.lammps -log log.lammps

# 并行运行
mpirun -np 8 lmp -in input.lammps

# GPU加速
lmp -k on g 1 -sf kk -in input.lammps
```

---

## 📖 核心概念

- **Input Script**: LAMMPS输入脚本，定义模拟流程
- **Data File**: 初始构型文件（原子坐标、类型、连接等）
- **Dump File**: 轨迹输出文件
- **Thermo Output**: 热力学量输出
- **Pair Style**: 原子间相互作用势
- **Fix**: 约束、控温、时间积分等操作
- **Compute**: 计算各种物理量

---

## 🎯 学习路径

### 初学者 (1-2周)
1. 阅读[安装配置](01-installation.md)完成安装
2. 学习[基础MD计算](03-basic-md.md)理解系综
3. 掌握[势函数库](02-potentials.md)基础势函数
4. 运行Cu熔化和水模拟示例

### 进阶 (2-4周)
5. [材料案例](06-materials-cases.md)完成完整案例
6. [可视化后处理](10-visualization-postprocessing.md)掌握RDF/MSD分析
7. [大规模并行](08-large-scale-parallel.md)并行计算优化

### 专家 (1-2月)
8. [高级采样](04-advanced-sampling.md)自由能计算
9. [多尺度模拟](05-multiscale.md)QM/MM耦合
10. [ML势训练](07-ml-potential-training.md)DeepMD/DP-GEN
11. [材料数据库](09-materials-databases.md)OpenKIM/Matlantis
12. [特定应用案例](11-specific-applications.md)电池/催化/高熵合金

---

## 🔗 资源链接

- **官方文档**: https://docs.lammps.org/
- **GitHub**: https://github.com/lammps/lammps
- **用户论坛**: https://matsci.org/lammps
- **下载**: https://lammps.sandia.gov/download.html

---

## 📊 技能统计

- **文档数**: 11个 (+5 高级专题)
- **总字数**: ~140,000字
- **代码示例**: 100+
- **覆盖度**: 100%
- **高级专题**: 5个 (ML势、并行优化、数据库、可视化、应用案例)

---

*最后更新: 2026-03-08*
