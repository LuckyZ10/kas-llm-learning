# 01 - 15分钟快速入门 | 15-Minute Quick Start

> **学习目标**: 在15分钟内完成第一个DFT+ML+MD工作流  
> **Learning Goal**: Complete your first DFT+ML+MD workflow in 15 minutes

---

## 📋 目录 | Table of Contents

1. [环境准备 | Environment Setup](#1-环境准备--environment-setup)
2. [快速运行 | Quick Run](#2-快速运行--quick-run)
3. [理解工作流 | Understanding the Workflow](#3-理解工作流--understanding-the-workflow)
4. [查看结果 | Viewing Results](#4-查看结果--viewing-results)
5. [常见错误 | Common Errors](#5-常见错误--common-errors)
6. [练习题 | Exercises](#6-练习题--exercises)

---

## 1. 环境准备 | Environment Setup

### 1.1 安装依赖 | Install Dependencies

```bash
# 创建conda环境 | Create conda environment
conda create -n dft-lammps python=3.10 -y
conda activate dft-lammps

# 安装核心包 | Install core packages
pip install ase pymatgen dpdata numpy pandas

# 安装DeePMD-kit | Install DeePMD-kit (optional for this tutorial)
pip install deepmd-kit

# 设置Materials Project API密钥 | Set MP API key
export MP_API_KEY="your_api_key_here"
```

### 1.2 验证安装 | Verify Installation

```bash
python -c "import ase; import pymatgen; print('✓ All packages installed!')"
```

---

## 2. 快速运行 | Quick Run

### 2.1 下载示例代码 | Download Example Code

```bash
cd /root/.openclaw/workspace/dft_lammps_research

# 复制快速入门示例 | Copy quick start example
cp examples/quick_start/simple_workflow.py ./my_first_workflow.py
```

### 2.2 运行最小工作流 | Run Minimal Workflow

```python
#!/usr/bin/env python3
"""
最小工作流示例 | Minimal Workflow Example
"""
from integrated_materials_workflow import IntegratedMaterialsWorkflow
from integrated_materials_workflow import (
    IntegratedWorkflowConfig,
    MaterialsProjectConfig,
    DFTStageConfig,
    MLPotentialConfig,
    MDStageConfig,
    AnalysisConfig
)

# Step 1: 配置工作流 | Configure workflow
config = IntegratedWorkflowConfig(
    workflow_name="quick_start_demo",
    working_dir="./output_quick_start",
    mp_config=MaterialsProjectConfig(
        api_key=None,  # 从环境变量读取 | Read from environment
        max_entries=10
    ),
    dft_config=DFTStageConfig(
        code="vasp",  # 或 "espresso"
        encut=400,    # 为速度降低截断能 | Lower cutoff for speed
        ncores=4
    ),
    ml_config=MLPotentialConfig(
        framework="deepmd",
        preset="fast",  # 快速训练 | Fast training
        num_models=2    # 减少模型数量 | Fewer models
    ),
    md_config=MDStageConfig(
        temperatures=[300, 500],  # 减少温度点 | Fewer temperatures
        nsteps_equil=1000,        # 减少平衡步数 | Shorter equilibration
        nsteps_prod=5000          # 减少生产步数 | Shorter production
    )
)

# Step 2: 创建并运行工作流 | Create and run workflow
workflow = IntegratedMaterialsWorkflow(config)

# 从Materials Project获取Li3PS4结构 | Get Li3PS4 structure from MP
results = workflow.run(formula="Li3PS4")

# 输出结果 | Print results
print("\n" + "="*60)
print("工作流完成! | Workflow Completed!")
print("="*60)
print(f"化学式 | Formula: {results['formula']}")
print(f"DFT能量 | DFT Energy: {results['dft']['energy_per_atom']:.4f} eV/atom")
print(f"扩散系数 | Diffusion Coeff: {results['analysis']['diffusion_coefficients']}")
print(f"活化能 | Activation Energy: {results['analysis']['activation_energy']:.3f} eV")
```

### 2.3 执行命令 | Execute Command

```bash
# 提交到本地计算 | Run locally
python my_first_workflow.py

# 或使用HPC调度 | Or submit to HPC
sbatch -J quick_start --wrap="python my_first_workflow.py"
```

**预期输出 | Expected Output:**
```
============================================================
Starting Integrated Materials Workflow: quick_start_demo
============================================================

============================================================
Stage 1/5: fetch_structure
============================================================
  [10.0%] Searching for formula: Li3PS4
  [100.0%] Retrieved 1 structures
  Stage completed: success (took 2.3s)

============================================================
Stage 2/5: dft_calculation
============================================================
  [30.0%] Running structure relaxation
  [100.0%] DFT calculation completed
  Stage completed: success (took 180.5s)

...

============================================================
Workflow completed in 1250.3s
============================================================

化学式 | Formula: Li3P1S4
DFT能量 | DFT Energy: -4.8234 eV/atom
扩散系数 | Diffusion Coeff: {300: 1.2e-06, 500: 5.8e-05}
活化能 | Activation Energy: 0.312 eV
```

---

## 3. 理解工作流 | Understanding the Workflow

### 3.1 工作流架构 | Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    输入 | Input                             │
│              化学式 / MP ID / 结构文件                       │
│         Formula / MP ID / Structure File                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: 获取结构 | Fetch Structure                        │
│  • 从Materials Project下载                                  │
│  • Download from Materials Project                          │
│  • 或读取本地文件 | Or read local file                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: DFT计算 | DFT Calculation                         │
│  • 结构优化 (VASP/QE)                                        │
│  • Structure Relaxation                                     │
│  • 生成训练数据                                             │
│  • Generate Training Data                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: ML势训练 | ML Potential Training                  │
│  • DeePMD-kit / NEP / MACE                                  │
│  • 模型集成 (ensemble)                                      │
│  • Model Ensemble                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: MD模拟 | MD Simulation                            │
│  • LAMMPS + ML势                                            │
│  • LAMMPS + ML Potential                                    │
│  • 多温度采样                                               │
│  • Multi-temperature Sampling                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: 分析 | Analysis                                   │
│  • 扩散系数计算                                             │
│  • Diffusion Coefficient                                    │
│  • Arrhenius拟合                                            │
│  • Arrhenius Fitting                                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   输出 | Output                             │
│            报告 + 模型 + 轨迹文件                            │
│       Report + Models + Trajectory Files                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 关键类说明 | Key Classes Explained

| 类名 | Class Name | 功能 | Function |
|------|-----------|------|----------|
| `IntegratedWorkflowConfig` | 工作流配置 | Workflow configuration | 配置所有阶段参数 |
| `StructureFetcher` | 结构获取器 | Structure fetcher | 从MP或文件获取结构 |
| `DFTStage` | DFT计算阶段 | DFT calculation stage | 运行VASP/QE计算 |
| `MLTrainingStage` | ML训练阶段 | ML training stage | 训练势函数模型 |
| `MDSimulationStage` | MD模拟阶段 | MD simulation stage | 运行LAMMPS模拟 |
| `AnalysisStage` | 分析阶段 | Analysis stage | 计算物理性质 |

---

## 4. 查看结果 | Viewing Results

### 4.1 输出文件结构 | Output Directory Structure

```
output_quick_start/
├── workflow_report.json          # 完整报告 | Full report
├── integrated_workflow.log       # 运行日志 | Execution log
├── initial_structure.vasp        # 初始结构 | Initial structure
├── dft_results/                  # DFT结果
│   ├── CONTCAR                   # 优化后结构
│   ├── OUTCAR                    # VASP输出
│   └── results.json              # 解析结果
├── ml_models/                    # ML模型
│   ├── model_0/graph.pb          # 模型0 (冻结)
│   ├── model_1/graph.pb          # 模型1 (冻结)
│   └── input.json                # 训练输入
├── md_results/                   # MD结果
│   ├── T300/                     # 300K模拟
│   │   ├── dump.lammpstrj        # 轨迹文件
│   │   └── in.lammps             # LAMMPS输入
│   └── T500/                     # 500K模拟
└── analysis/                     # 分析结果
    ├── analysis_results.json     # 扩散系数等
    └── arrhenius_plot.png        # Arrhenius图
```

### 4.2 阅读报告 | Reading the Report

```python
import json

# 加载报告 | Load report
with open('output_quick_start/workflow_report.json', 'r') as f:
    report = json.load(f)

# 查看结果 | View results
print(f"总耗时 | Total time: {report['progress']['total_elapsed']:.1f}s")
print(f"DFT能量 | DFT energy: {report['results']['dft_energy']:.4f} eV/atom")

# 分析结果
analysis = report['results']['analysis']
print(f"活化能 | Activation Energy: {analysis['activation_energy']:.3f} eV")

# 查看各阶段耗时
for stage in report['progress']['history']:
    print(f"{stage['stage']}: {stage['elapsed']:.1f}s - {stage['status']}")
```

---

## 5. 常见错误 | Common Errors

### 5.1 错误速查表 | Error Quick Reference

| 错误信息 | Error Message | 原因 | Cause | 解决方案 | Solution |
|----------|--------------|------|-------|----------|----------|
| `MP_API_KEY not found` | MP API key not found | 未设置API密钥 | API key not set | `export MP_API_KEY=xxx` | Set environment variable |
| `VASP command not found` | VASP command not found | VASP未安装或未在PATH中 | VASP not installed | 检查VASP安装 | Check VASP installation |
| `OUTCAR not found` | OUTCAR not found | DFT计算失败 | DFT failed | 检查计算资源 | Check compute resources |
| `dp command not found` | dp command not found | DeePMD未安装 | DeePMD not installed | `pip install deepmd-kit` | Install DeePMD |
| `LAMMPS error` | LAMMPS error | LAMMPS输入错误 | LAMMPS input error | 检查in.lammps | Check LAMMPS input |
| `Memory error` | Memory error | 内存不足 | Out of memory | 减少核心数 | Reduce ncores |

### 5.2 调试技巧 | Debugging Tips

```python
# 启用详细日志 | Enable verbose logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# 跳过某些阶段进行调试 | Skip stages for debugging
config.stages["dft_calculation"].enabled = False  # 使用已有DFT结果
config.stages["ml_training"].enabled = False      # 使用已有模型

# 单步执行 | Step-by-step execution
workflow.fetcher.fetch_from_mp(formula="Li3PS4")  # 只获取结构
workflow.dft_stage.run_relaxation(structure, "./test_dft")  # 只运行DFT
```

---

## 6. 练习题 | Exercises

### 练习 1: 修改材料 | Exercise 1: Change Material

```python
# 修改化学式 | Change the formula
results = workflow.run(formula="Li2S")  # 尝试其他材料

# 或使用MP ID | Or use MP ID
results = workflow.run(material_id="mp-1138")
```

### 练习 2: 禁用阶段 | Exercise 2: Disable Stages

```python
# 跳过DFT阶段 (使用预计算数据) | Skip DFT (use precomputed data)
config.stages["dft_calculation"].enabled = False

# 跳过ML训练 (使用预训练模型) | Skip ML training
config.stages["ml_training"].enabled = False
config.stages["md_simulation"].enabled = True
```

### 练习 3: 调整参数 | Exercise 3: Adjust Parameters

```python
# 增加温度点 | Add more temperatures
config.md_config.temperatures = [300, 400, 500, 600, 700, 800, 900]

# 提高DFT精度 | Increase DFT accuracy
config.dft_config.encut = 600
config.dft_config.kpoints_density = 0.2

# 更长的MD模拟 | Longer MD simulation
config.md_config.nsteps_prod = 100000
```

### 练习 4: 自定义分析 | Exercise 4: Custom Analysis

```python
# 访问原始数据 | Access raw data
analysis_stage = workflow.analysis_stage

# 计算特定原子的扩散 | Compute diffusion for specific atom
D = analysis_stage.analyze_diffusion(
    trajectory_file="output_quick_start/md_results/T300/dump.lammpstrj",
    atom_type="Li",
    timestep=1.0
)
print(f"Li扩散系数 | Li diffusion coefficient: {D:.2e} cm²/s")
```

---

## 🎉 恭喜! | Congratulations!

你已完成第一个DFT+ML+MD工作流!接下来的教程将深入每个阶段。

**下一步**: [02 - DFT计算基础教程](02_dft_basics.md)

---

## 📚 参考资源 | References

- [ASE Documentation](https://wiki.fysik.dtu.dk/ase/)
- [Pymatgen Documentation](https://pymatgen.org/)
- [DeePMD-kit Documentation](https://deepmd.readthedocs.io/)
- [LAMMPS Manual](https://docs.lammps.org/)

---

## 📝 术语表 | Glossary

| 中文 | English | 说明 |
|------|---------|------|
| DFT | Density Functional Theory | 密度泛函理论 |
| ML势 | ML Potential | 机器学习势函数 |
| MD | Molecular Dynamics | 分子动力学 |
| AIMD | Ab Initio MD | 从头算分子动力学 |
| 扩散系数 | Diffusion Coefficient | 描述离子扩散速率 |
| 活化能 | Activation Energy | 扩散能垒 |
| 系综 | Ensemble | NVT, NPT等热力学系综 |
