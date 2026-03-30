# LAMMPS 快速参考索引

## 按主题查找

### 入门
- [SKILL.md](SKILL.md) - 技能主文档
- [01-installation.md](01-installation.md) - 安装配置
- [03-basic-md.md](03-basic-md.md) - 基础MD计算

### 势函数
- [02-potentials.md](02-potentials.md) - 势函数总览
- [02-potentials.md#eammeam势](02-potentials.md) - EAM/MEAM金属势
- [02-potentials.md#机器学习势](02-potentials.md) - ACE/DeepMD/NequIP
- [02-potentials.md#reaxff反应力场](02-potentials.md) - ReaxFF

### 采样方法
- [04-advanced-sampling.md](04-advanced-sampling.md) - 增强采样总览
- [04-advanced-sampling.md#伞形采样](04-advanced-sampling.md) - Umbrella Sampling
- [04-advanced-sampling.md#副本交换分子动力学](04-advanced-sampling.md) - REMD
- [04-advanced-sampling.md#metadynamics](04-advanced-sampling.md) - Metadynamics
- [04-advanced-sampling.md#温度加速分子动力学](04-advanced-sampling.md) - TAD

### 多尺度
- [05-multiscale.md](05-multiscale.md) - 多尺度方法
- [05-multiscale.md#qmmm耦合](05-multiscale.md) - QM/MM
- [05-multiscale.md#机器学习势多尺度](05-multiscale.md) - DeepMD工作流

### 材料案例
- [06-materials-cases.md](06-materials-cases.md) - 案例总览
- [06-materials-cases.md#金属体系](06-materials-cases.md) - Cu/Ni-Al/位错
- [06-materials-cases.md#聚合物体系](06-materials-cases.md) - PE/PS/纳米复合
- [06-materials-cases.md#生物分子体系](06-materials-cases.md) - 蛋白质/膜/DNA
- [06-materials-cases.md#界面体系](06-materials-cases.md) - 金属-水/油-水/接触线

### 分析工具
- [07-analysis.md](07-analysis.md) - 分析总览
- [07-analysis.md#径向分布函数rdf](07-analysis.md) - RDF计算
- [07-analysis.md#均方位移msd与扩散系数](07-analysis.md) - MSD/Diffusion
- [07-analysis.md#自由能计算](07-analysis.md) - WHAM/MBAR
- [07-analysis.md#python后处理](07-analysis.md) - Python分析

### 性能优化
- [08-performance.md](08-performance.md) - 优化总览
- [08-performance.md#mpi并行优化](08-performance.md) - MPI
- [08-performance.md#gpu加速](08-performance.md) - Kokkos GPU
- [08-performance.md#负载均衡](08-performance.md) - Load Balancing

### 高级专题
- [09-advanced-topics.md](09-advanced-topics.md) - 专家技巧
- [09-advanced-topics.md#反应力场模拟](09-advanced-topics.md) - ReaxFF
- [09-advanced-topics.md#相变与成核](09-advanced-topics.md) - 成核模拟
- [09-advanced-topics.md#缺陷演化模拟](09-advanced-topics.md) - 缺陷分析

### 工具集成
- [10-integrations.md](10-integrations.md) - Python/ASE/OVITO集成
- [10-integrations.md#python接口](10-integrations.md) - PyLAMMPS
- [10-integrations.md#ase集成](10-integrations.md) - ASE
- [10-integrations.md#ovito编程](10-integrations.md) - OVITO

## 按计算类型查找

| 计算类型 | 文档 | 关键命令 |
|---------|------|---------|
| NVT平衡 | [03-basic-md.md](03-basic-md.md) | `fix nvt` |
| NPT平衡 | [03-basic-md.md](03-basic-md.md) | `fix npt` |
| 能量最小化 | [03-basic-md.md](03-basic-md.md) | `minimize` |
| 拉伸变形 | [06-materials-cases.md](06-materials-cases.md) | `fix deform` |
| 伞形采样 | [04-advanced-sampling.md](04-advanced-sampling.md) | `fix colvars` |
| 副本交换 | [04-advanced-sampling.md](04-advanced-sampling.md) | `fix temper` |
| QM/MM | [05-multiscale.md](05-multiscale.md) | `fix qmmm` |
| RDF分析 | [07-analysis.md](07-analysis.md) | `compute rdf` |
| MSD分析 | [07-analysis.md](07-analysis.md) | `compute msd` |

## 命令速查表

### 初始化命令
```
units metal/real/lj
atom_style atomic/full/charge
boundary p p p / p s p / f f p
read_data / read_restart / lattice + create_atoms
```

### 势函数命令
```
pair_style lj/cut, eam/alloy, reax/c, sw, tersoff
pair_coeff * * potential.file Element1 Element2
pair_modify mix arithmetic / geometric
```

### 约束命令
```
fix nvt, fix npt, fix nve
fix langevin, fix temp/berendsen
fix spring/self, fix spring/couple
fix wall/*, fix indent
```

### 输出命令
```
dump custom, dump atom, dump netcdf
thermo_style custom step temp pe press
restart 1000 file.restart
write_dump / write_restart / write_data
```

### 计算命令
```
compute pe/atom, ke/atom, stress/atom
compute rdf, msd, vacf
compute cna/atom, cna/atom
```

## 外部工具链接

### 可视化
- [VMD](https://www.ks.uiuc.edu/Research/vmd/) - 生物分子可视化
- [OVITO](https://www.ovito.org/) - 材料可视化
- [ParaView](https://www.paraview.org/) - 通用可视化

### 分析
- [MDAnalysis](https://www.mdanalysis.org/) - Python MD分析
- [MDTraj](https://mdtraj.org/) - 轨迹分析
- [Freud](https://freud.readthedocs.io/) - 粒子分析

### 其他MD软件
- [GROMACS](http://www.gromacs.org/) - 生物分子MD
- [Amber](https://ambermd.org/) - 生物分子MD
- [NAMD](https://www.ks.uiuc.edu/Research/namd/) - 大规模生物MD
- [OpenMM](http://openmm.org/) - GPU加速MD
