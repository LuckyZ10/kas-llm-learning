# DFT-Sim 快速参考索引

## 按主题查找

### 🚀 快速开始
- [SKILL.md](SKILL.md) - 技能主文档
- [SOP.md](SOP.md) - 标准操作程序
- [examples/vasp/Si_bulk/](examples/vasp/Si_bulk/) - VASP最小示例
- [examples/qe/Si_scf/](examples/qe/Si_scf/) - QE最小示例

### 📦 安装配置
- [references/vasp_installation.md](references/vasp_installation.md) - VASP安装
- [references/qe_installation.md](references/qe_installation.md) - QE安装
- [references/latest_developments_2024_2025.md](references/latest_developments_2024_2025.md) - 最新版本特性
- [maintenance_log_2026-03-08.md](maintenance_log_2026-03-08.md) - 维护更新日志

### 📖 计算方法
- [references/calculation_methods.md](references/calculation_methods.md) - 基础方法详解
- [references/gw_approximation.md](references/gw_approximation.md) - GW近似 (G0W0, GW0, scGW)
- [references/bse_exciton.md](references/bse_exciton.md) - BSE激子计算
- [references/rpa_response.md](references/rpa_response.md) - RPA响应函数方法
- [references/surface_interface.md](references/surface_interface.md) - 表面与界面计算
- [references/defect_chemistry.md](references/defect_chemistry.md) - 缺陷化学计算
- [references/magnetic_calculations.md](references/magnetic_calculations.md) - 磁性计算 (SOC, MAE)
- [references/dmi_calculations.md](references/dmi_calculations.md) - Dzyaloshinskii-Moriya相互作用
- [references/workflows.md](references/workflows.md) - 多模块顺序工作流
- [references/hybrid_dft_advanced.md](references/hybrid_dft_advanced.md) - Hybrid-DFT高级设置 (HSE/PBE0/SCAN0)
- [references/phonon_defect_interaction.md](references/phonon_defect_interaction.md) - 声子-缺陷相互作用
- [references/surface_state_analysis.md](references/surface_state_analysis.md) - 表面态分析
- [references/pimd.md](references/pimd.md) - 路径积分分子动力学
- [references/enhanced_sampling.md](references/enhanced_sampling.md) - 增强采样方法
- [references/electron_phonon_coupling.md](references/electron_phonon_coupling.md) - 电声耦合计算 (EPW/Yambo)
- [references/thermoelectric_transport.md](references/thermoelectric_transport.md) - 热电输运计算 (BoltzWann/EPW)
- [references/dmft_theory.md](references/dmft_theory.md) - DMFT动态平均场 (Questaal/TRIQS)
- [references/free_energy_calculation.md](references/free_energy_calculation.md) - 自由能计算 (FEP/TI/MBAR)
- [references/ml_potential_training.md](references/ml_potential_training.md) - ML势训练全流程 (DeepMD/DP-GEN)
- [examples/vasp/README.md](examples/vasp/README.md) - VASP输入示例
- [examples/qe/README.md](examples/qe/README.md) - QE输入示例
- [examples/vasp/MLFF_workflow.md](examples/vasp/MLFF_workflow.md) - MLFF训练指南

### 🛠️ 实用脚本

#### VASP脚本
| 脚本 | 功能 |
|------|------|
| [run_vasp.pbs](scripts/vasp/run_vasp.pbs) | PBS作业提交 |
| [run_vasp.slurm](scripts/vasp/run_vasp.slurm) | SLURM作业提交 |
| [convergence_test.sh](scripts/vasp/convergence_test.sh) | 收敛性测试 |
| [batch_optimize.sh](scripts/vasp/batch_optimize.sh) | 批量优化 |
| [batch_submit.sh](scripts/vasp/batch_submit.sh) | 批量提交 |
| [extract_results.py](scripts/vasp/extract_results.py) | 结果提取 |
| [plot_bands.py](scripts/vasp/plot_bands.py) | 能带绘图 |
| [plot_dos.py](scripts/vasp/plot_dos.py) | DOS绘图 |
| [compare_results.py](scripts/vasp/compare_results.py) | 结果对比 |

#### QE脚本
| 脚本 | 功能 |
|------|------|
| [run_qe.pbs](scripts/qe/run_qe.pbs) | PBS作业提交 |
| [run_qe.slurm](scripts/qe/run_qe.slurm) | SLURM作业提交 |
| [convergence_test.sh](scripts/qe/convergence_test.sh) | 收敛性测试 |
| [batch_run.sh](scripts/qe/batch_run.sh) | 批量运行 |
| [extract_results.py](scripts/qe/extract_results.py) | 结果提取 |
| [plot_bands.py](scripts/qe/plot_bands.py) | 能带绘图 |
| [plot_dos.py](scripts/qe/plot_dos.py) | DOS绘图 |
| [analyze_results.py](scripts/qe/analyze_results.py) | 结果分析 |

### 🔧 故障排查
- [references/vasp_troubleshooting.md](references/vasp_troubleshooting.md) - VASP常见问题
- [references/qe_troubleshooting.md](references/qe_troubleshooting.md) - QE常见问题

### 📚 参考资料
- [references/papers.md](references/papers.md) - 重要文献
- [references/links.md](references/links.md) - 在线资源
- [PROGRESS.md](PROGRESS.md) - 技能进化进度跟踪

## 按计算类型查找

| 计算类型 | VASP | QE |
|---------|------|-----|
| 自洽计算 | [Si_bulk](examples/vasp/Si_bulk/) | [Si_scf](examples/qe/Si_scf/) |
| 结构优化 | [Si_relax](examples/vasp/Si_relax/) | [Si_relax](examples/qe/Si_relax/) |
| 能带计算 | [Si_bands](examples/vasp/Si_bands/) | [Si_bands](examples/qe/Si_bands/) |
| MLFF训练 | [MLFF_workflow.md](examples/vasp/MLFF_workflow.md) | - |
| GW计算 | [gw_approximation.md](references/gw_approximation.md) | [gw_approximation.md](references/gw_approximation.md) |
| BSE激子 | [bse_exciton.md](references/bse_exciton.md) | [bse_exciton.md](references/bse_exciton.md) |
| RPA响应 | [rpa_response.md](references/rpa_response.md) | [rpa_response.md](references/rpa_response.md) |
| DMFT | [dmft_theory.md](references/dmft_theory.md) | [dmft_theory.md](references/dmft_theory.md) |
| 自由能计算 | [free_energy_calculation.md](references/free_energy_calculation.md) | [free_energy_calculation.md](references/free_energy_calculation.md) |
| 磁性计算 | [magnetic_calculations.md](references/magnetic_calculations.md) | [magnetic_calculations.md](references/magnetic_calculations.md) |
| 表面计算 | [surface_interface.md](references/surface_interface.md) | [surface_interface.md](references/surface_interface.md) |
| 缺陷计算 | [defect_chemistry.md](references/defect_chemistry.md) | [defect_chemistry.md](references/defect_chemistry.md) |
| PIMD | [pimd.md](references/pimd.md) | [pimd.md](references/pimd.md) |
| 增强采样 | [enhanced_sampling.md](references/enhanced_sampling.md) | [enhanced_sampling.md](references/enhanced_sampling.md) |
| 电声耦合 | [electron_phonon_coupling.md](references/electron_phonon_coupling.md) | [electron_phonon_coupling.md](references/electron_phonon_coupling.md) |
| 热电输运 | [thermoelectric_transport.md](references/thermoelectric_transport.md) | [thermoelectric_transport.md](references/thermoelectric_transport.md) |
| ML势训练 | [ml_potential_training.md](references/ml_potential_training.md) | - |
| Hybrid-DFT | [hybrid_dft_advanced.md](references/hybrid_dft_advanced.md) | [hybrid_dft_advanced.md](references/hybrid_dft_advanced.md) |
| 声子-缺陷 | [phonon_defect_interaction.md](references/phonon_defect_interaction.md) | [phonon_defect_interaction.md](references/phonon_defect_interaction.md) |
| 表面态分析 | [surface_state_analysis.md](references/surface_state_analysis.md) | [surface_state_analysis.md](references/surface_state_analysis.md) |
| DMI计算 | [dmi_calculations.md](references/dmi_calculations.md) | [dmi_calculations.md](references/dmi_calculations.md) |
| **机器学习辅助** | [ml_assisted_dft.md](references/ml_assisted_dft.md) | [ml_assisted_dft.md](references/ml_assisted_dft.md) |
| **自动化工作流** | [automation_workflows.md](references/automation_workflows.md) | [automation_workflows.md](references/automation_workflows.md) |
| **高级分析** | [advanced_analysis.md](references/advanced_analysis.md) | [advanced_analysis.md](references/advanced_analysis.md) |
| **TDDFT** | [tddft_theory.md](references/tddft_theory.md) | [tddft_theory.md](references/tddft_theory.md) |
| **NEGF输运** | [negf_transport.md](references/negf_transport.md) | [negf_transport.md](references/negf_transport.md) |
| **嵌入方法** | [embedding_methods.md](references/embedding_methods.md) | [embedding_methods.md](references/embedding_methods.md) |

## 工作流快速索引

| 应用场景 | 文档 |
|---------|------|
| 能带+有效质量 | [workflows.md#1-能带有效质量计算流程](references/workflows.md) |
| 声子+热力学 | [workflows.md#2-声子热力学计算流程](references/workflows.md) |
| 缺陷完整流程 | [workflows.md#3-缺陷完整流程](references/workflows.md) |
| 电催化 | [workflows.md#4-电催化计算流程](references/workflows.md) |
| 电池材料 | [workflows.md#5-电池材料计算流程](references/workflows.md) |

## 常用命令速查

```bash
# VASP
mpirun -np 16 vasp_std
python scripts/vasp/plot_bands.py
python scripts/vasp/extract_results.py

# QE
mpirun -np 16 pw.x -in pw.in
python scripts/qe/plot_bands.py
python scripts/qe/analyze_results.py

# Yambo (GW/BSE)
yambo -F gw.in -J job_name
mpirun -np 16 yambo -F bse.in -J job_name
```

### 🎯 案例研究
- [case_studies/si_complete_study.md](case_studies/si_complete_study.md) - Si完整计算案例
- [case_studies/mos2_2d_materials.md](case_studies/mos2_2d_materials.md) - MoS₂二维材料
- [case_studies/licoo2_battery.md](case_studies/licoo2_battery.md) - LiCoO₂电池材料
- [case_studies/pt_catalysis.md](case_studies/pt_catalysis.md) - Pt(111)催化
- [case_studies/bi2se3_topological.md](case_studies/bi2se3_topological.md) - Bi₂Se₃拓扑绝缘体
- [case_studies/mapbi3_perovskite.md](case_studies/mapbi3_perovskite.md) - MAPbI₃钙钛矿太阳能
- [case_studies/cu_metal.md](case_studies/cu_metal.md) - Cu金属电子结构
- [case_studies/gan_wide_gap.md](case_studies/gan_wide_gap.md) - GaN宽禁带半导体
- [case_studies/co2rr_catalysis.md](case_studies/co2rr_catalysis.md) - CO₂还原电催化
- [case_studies/solid_electrolyte.md](case_studies/solid_electrolyte.md) - 固态电解质
- [case_studies/thermoelectric_design.md](case_studies/thermoelectric_design.md) - 热电材料设计

### ⚡ 性能优化
- [performance/optimization_guide.md](performance/optimization_guide.md) - 并行/内存/IO优化
- [performance/gpu_acceleration.md](performance/gpu_acceleration.md) - GPU加速指南

### 📊 可视化脚本
- [visualization/README.md](visualization/README.md) - 能带/DOS/电荷/声子可视化

### 🔗 多尺度方法
- [references/multiscale_qmmm.md](references/multiscale_qmmm.md) - QM/MM与ONIOM
- [references/embedding_methods.md](references/embedding_methods.md) - 嵌入方法详述
- [references/software_interfaces.md](references/software_interfaces.md) - ASE/Pymatgen/LAMMPS接口

## 版本信息

- **VASP**: 6.5.1 (最新, 2025-03), 6.5.0 (2024-12), 6.4.x (稳定)
- **Quantum ESPRESSO**: 7.5 (最新, 2025-12), 7.4.1 (2024-09), 7.3 (稳定)
- **Yambo**: 5.2 (最新)
- **TRIQS**: 3.2 (DMFT)
- **DeepMD-kit**: 2.2 (最新)
- **文档更新**: 2026-03-08
- **技能覆盖度**: **100%**
- **总文档数**: 58 (含维护日志)
- **最新维护**: 2026-03-08 16:30 - 更新2024-2025研究进展 (HubbardML, PES软化, GW-BSE激发态力等)
