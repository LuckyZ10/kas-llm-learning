# DFT-Sim: 第一性原理计算技能

## 简介

本技能涵盖两种主流的第一性原理计算软件：
- **VASP** (Vienna Ab initio Simulation Package) - 商业软件，功能强大
- **Quantum ESPRESSO** - 开源免费，社区活跃

## 目录结构

```
dft-sim/
├── SKILL.md              # 本文件 - 技能主文档
├── SOP.md               # 标准操作程序
├── scripts/              # 实用脚本
│   ├── vasp/            # VASP相关脚本
│   │   ├── run_vasp.pbs
│   │   ├── run_vasp.slurm
│   │   ├── convergence_test.sh
│   │   ├── batch_optimize.sh
│   │   ├── batch_submit.sh
│   │   ├── extract_results.py
│   │   ├── plot_bands.py
│   │   ├── plot_dos.py
│   │   └── compare_results.py
│   └── qe/              # QE相关脚本
│       ├── run_qe.pbs
│       ├── run_qe.slurm
│       ├── convergence_test.sh
│       ├── batch_run.sh
│       ├── extract_results.py
│       ├── plot_bands.py
│       ├── plot_dos.py
│       └── analyze_results.py
├── examples/            # 输入文件示例
│   ├── vasp/
│   │   ├── README.md
│   │   ├── MLFF_workflow.md
│   │   ├── Si_bulk/     # 完整示例
│   │   ├── Si_relax/
│   │   └── Si_bands/
│   └── qe/
│       ├── README.md
│       ├── Si_scf/
│       ├── Si_relax/
│       └── Si_bands/
└── references/         # 参考文献和链接 (33个文档)
    ├── vasp_installation.md
    ├── qe_installation.md
    ├── calculation_methods.md
    ├── latest_developments_2024_2025.md
    ├── vasp_troubleshooting.md
    ├── qe_troubleshooting.md
    ├── gw_approximation.md
    ├── bse_exciton.md
    ├── rpa_response.md
    ├── dmft_theory.md
    ├── hybrid_dft_advanced.md
    ├── electron_phonon_coupling.md
    ├── thermoelectric_transport.md
    ├── phonon_defect_interaction.md
    ├── surface_interface.md
    ├── surface_state_analysis.md
    ├── defect_chemistry.md
    ├── magnetic_calculations.md
    ├── dmi_calculations.md
    ├── pimd.md
    ├── enhanced_sampling.md
    ├── free_energy_calculation.md
    ├── ml_potential_training.md
    ├── workflows.md
    ├── papers.md
    └── links.md
```

## 软件对比

| 特性 | VASP | Quantum ESPRESSO |
|------|------|------------------|
| 许可证 | 商业软件 | 开源GPL |
| 基组 | 平面波+PAW | 平面波+USPP/PAW/NCPP |
| GPU支持 | 部分支持 | 7.0+版本全面支持 |
| 机器学习力场 | MLFF (6.4+) | 需外部接口 |
| 社区支持 | 官方论坛 | GitHub/GitLab |

## 快速开始

### VASP

1. **获取许可证**: 通过 VASP Software GmbH 申请学术/商业许可
2. **下载源码**: 从 VASP Portal 下载
3. **编译安装**: 见 [VASP安装指南](#vasp安装配置)

### Quantum ESPRESSO

```bash
# 下载最新版本 (7.4)
wget https://gitlab.com/QEF/q-e/-/archive/qe-7.4/q-e-qe-7.4.tar.gz
tar -xzf q-e-qe-7.4.tar.gz
cd q-e-qe-7.4

# 编译安装
mkdir build && cd build
../configure --enable-openmp
make all -j$(nproc)
make install
```

## 核心计算方法

### 1. 结构优化 (Geometry Optimization)

寻找体系的能量最低结构，是所有计算的基础。

### 2. 能带计算 (Band Structure)

计算电子能带结构，分析材料的导电性质。

### 3. 态密度 (Density of States)

- **总态密度 (TDOS)**: 整个体系的电子态分布
- **分波态密度 (PDOS)**: 各原子/轨道的贡献

### 4. 分子动力学 (Molecular Dynamics)

- **从头算分子动力学 (AIMD)**: 基于DFT的MD
- **机器学习力场 (MLFF)**: VASP 6.4+ 支持

### 5. 声子计算 (Phonon)

- 密度泛函微扰理论 (DFPT)
- 有限位移法

### 6. 其他高级方法

- **DFT+U**: 强关联体系修正
- **杂化泛函**: HSE06, PBE0, SCAN0等 ([详细参数优化](references/hybrid_dft_advanced.md))
- **GW近似**: 准粒子能带计算 ([G0W0/GW0/scGW](references/gw_approximation.md))
- **BSE**: 激子性质计算 ([激子束缚能](references/bse_exciton.md))
- **DMFT**: 动态平均场理论 ([强关联体系](references/dmft_theory.md))
- **RPA**: 随机相近似 ([响应函数](references/rpa_response.md))

### 7. 分子动力学

- **从头算MD (AIMD)**: 标准DFT分子动力学
- **PIMD**: 路径积分分子动力学 ([量子核效应](references/pimd.md))
- **增强采样**: Metadynamics, ABF, 伞形采样 ([自由能面](references/enhanced_sampling.md))
- **ML力场**: DeepMD, M3GNet, CHGNet, MACE ([ML势训练](references/ml_potential_training.md))

### 8. 声子与输运

- **电声耦合**: EPW, Yambo接口 ([Eliashberg函数](references/electron_phonon_coupling.md))
- **热电输运**: BoltzWann, EPW计算 ([Seebeck系数/电导率](references/thermoelectric_transport.md))
- **声子-缺陷相互作用**: 局域声子模式, 振动熵 ([缺陷热力学](references/phonon_defect_interaction.md))

### 9. 表面、缺陷与磁性

- **表面态分析**: 层投影能带, 拓扑表面态 ([表面电子态](references/surface_state_analysis.md))
- **缺陷化学**: 形成能, 跃迁能级, 有限尺寸修正 ([缺陷计算](references/defect_chemistry.md))
- **DMI**: Dzyaloshinskii-Moriya相互作用 ([斯格明子计算](references/dmi_calculations.md))
- **磁性**: SOC, MAE, 非共线磁性 ([磁性计算](references/magnetic_calculations.md))

## 最新进展 (2024-2025)

### VASP 6.5 (2025年发布)

- **电子-声子耦合**: 零点能带重整化、输运系数
- **Python插件**: 通过C++接口与Python交互
- **BSE GPU加速**: Lanczos对角化和时间演化BSE
- **机器学习力场改进**: 引入溢出因子(spilling factor)作为误差估计
- **库仑核截断**: 开放边界条件用于偶极/带电分子、2D材料

### Quantum ESPRESSO 7.4 (2024年发布)

- **GPU加速全面支持**: CUDA和OpenACC优化
- **双化学势线性响应声子计算**
- **对称性检测改进**: 通过物种比较
- **性能提升**: A100 GPU上可达3倍加速

## 文件说明

- `scripts/` - 自动化脚本，包括作业提交、数据提取等
- `examples/` - 各类计算的输入文件模板
- `references/` - 重要文献和在线资源链接

## 使用建议

1. **初学者**: 从Quantum ESPRESSO开始，免费且文档丰富
2. **生产环境**: 根据机构许可选择VASP
3. **大规模计算**: 考虑QE的GPU加速版本
4. **机器学习**: VASP 6.4+的MLFF功能强大

## 参考资源

- VASP Wiki: https://www.vasp.at/wiki/
- QE官网: https://www.quantum-espresso.org/
- 本技能详细文档见各子目录
