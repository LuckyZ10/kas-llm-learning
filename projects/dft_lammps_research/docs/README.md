# DFT + LAMMPS 多尺度材料计算框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen.svg)](./tutorials/)

**English** | [中文](#概述)

A unified framework integrating Density Functional Theory (DFT) and Molecular Dynamics (MD) for multi-scale materials simulation, featuring Machine Learning potentials (DeepMD, NEP), active learning, and high-throughput screening capabilities.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🔬 **DFT Integration** | VASP and Quantum ESPRESSO interfaces with ASE |
| 🤖 **ML Potentials** | DeepMD-kit, NEP, and MACE training pipelines |
| 🔄 **Active Learning** | Automated Explore-Label-Retrain workflows |
| 🚀 **High-Throughput** | Batch calculations with workflow management |
| ⚡ **HPC Ready** | Slurm/PBS integration for cluster deployment |
| 📊 **Analysis Tools** | Diffusion, conductivity, and thermodynamic properties |

---

## 📖 Documentation

| Tutorial | Description |
|----------|-------------|
| [01 - Quick Start](tutorials/01_quick_start.md) | 15-minute hands-on introduction |
| [02 - DFT Basics](tutorials/02_dft_basics.md) | VASP/QE setup and convergence testing |
| [03 - ML Potential](tutorials/03_ml_potential.md) | Complete ML potential training guide |
| [04 - Active Learning](tutorials/04_active_learning.md) | Active learning workflows |
| [05 - High-Throughput](tutorials/05_high_throughput.md) | Screening case studies |
| [06 - HPC Deployment](tutorials/06_hpc_deployment.md) | Cluster usage guide |
| [07 - Advanced Workflows](tutorials/07_advanced_workflows.md) | Custom workflow development |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/dft-lammps-framework.git
cd dft-lammps-framework

# Create conda environment
conda create -n dft-lammps python=3.10
conda activate dft-lammps

# Install dependencies
pip install -r requirements.txt

# Install DFT/MD packages (optional)
pip install deepmd-kit  # ML potentials
conda install -c conda-forge lammps  # MD simulations
```

### Run Your First Workflow

```python
from integrated_materials_workflow import (
    IntegratedMaterialsWorkflow,
    IntegratedWorkflowConfig,
    MaterialsProjectConfig,
    DFTStageConfig,
    MLPotentialConfig
)

# Configure workflow
config = IntegratedWorkflowConfig(
    workflow_name="Li3PS4_demo",
    working_dir="./output",
    mp_config=MaterialsProjectConfig(api_key="your_mp_key"),
    dft_config=DFTStageConfig(code="vasp", encut=520),
    ml_config=MLPotentialConfig(framework="deepmd")
)

# Run end-to-end workflow
workflow = IntegratedMaterialsWorkflow(config)
results = workflow.run(formula="Li3PS4")

print(f"Activation Energy: {results['analysis']['activation_energy']:.3f} eV")
```

Or use the example script:

```bash
cd examples/quick_start
python simple_workflow.py
```

---

## 📁 Repository Structure

```
dft_lammps_research/
├── tutorials/                    # Step-by-step tutorials
│   ├── 01_quick_start.md
│   ├── 02_dft_basics.md
│   ├── 03_ml_potential.md
│   ├── 04_active_learning.md
│   ├── 05_high_throughput.md
│   ├── 06_hpc_deployment.md
│   └── 07_advanced_workflows.md
│
├── examples/                     # Example code and inputs
│   ├── quick_start/
│   ├── dft/
│   ├── ml_potential/
│   ├── active_learning/
│   └── high_throughput/
│
├── code_templates/               # Reusable code templates
│   ├── dft_workflow.py
│   ├── ml_potential_training.py
│   ├── md_simulation_lammps.py
│   └── active_learning_workflow.py
│
├── applications/                 # Application-specific workflows
│   ├── battery_screening_pipeline.py
│   └── Li3PS4_workflow_example.py
│
├── core_modules/                 # Core framework modules
│   ├── integrated_materials_workflow.py
│   ├── dft_to_lammps_bridge.py
│   ├── nep_training_pipeline.py
│   └── hpc_scheduler.py
│
└── README.md                     # This file
```

---

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DFT + LAMMPS Framework                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   DFT Stage │───▶│  ML Training│───▶│   MD Stage  │         │
│  │   (VASP/QE) │    │ (DeepMD/NEP)│    │  (LAMMPS)   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │            Active Learning Loop                      │      │
│  │     Explore → Label (DFT) → Retrain → Validate       │      │
│  └──────────────────────────────────────────────────────┘      │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │            High-Throughput Screening                 │      │
│  │  • Structure Generation                              │      │
│  │  • Batch Calculations                                │      │
│  │  • Property Analysis                                 │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💡 Usage Examples

### DFT Calculation

```python
from code_templates.dft_workflow import StructureOptimizer, DFTConfig

config = DFTConfig(code="vasp", functional="PBE", encut=520)
optimizer = StructureOptimizer(config)

atoms = optimizer.load_structure("POSCAR")
optimized = optimizer.relax_structure(fmax=0.01)
```

### ML Potential Training

```python
from code_templates.ml_potential_training import DeepMDTrainer, DeepMDConfig

config = DeepMDConfig(
    type_map=["Li", "P", "S"],
    descriptor_type="se_e2_a",
    rcut=6.0
)

trainer = DeepMDTrainer(config)
trainer.train()
model_path = trainer.freeze_model()
```

### Active Learning

```python
from code_templates.active_learning_workflow import ActiveLearningWorkflow

al_workflow = ActiveLearningWorkflow(
    initial_model="graph.pb",
    uncertainty_threshold=0.15
)

final_model = al_workflow.run(max_iterations=10)
```

---

## 🖥️ HPC Deployment

### Slurm Job Script

```bash
#!/bin/bash
#SBATCH -J dft_workflow
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 24:00:00
#SBATCH -p normal

module load vasp/6.3.0
module load deepmd-kit

python run_workflow.py --config workflow_config.yaml
```

### PBS Job Script

```bash
#!/bin/bash
#PBS -N dft_workflow
#PBS -l nodes=1:ppn=32
#PBS -l walltime=24:00:00

module load vasp
module load lammps

cd $PBS_O_WORKDIR
python run_workflow.py
```

---

## 📊 Benchmarks

| System | DFT (1 MD step) | ML Potential (1 MD step) | Speedup |
|--------|-----------------|--------------------------|---------|
| Li₃PS₄ (32 atoms) | 5 min | 0.1 s | 3000× |
| Li₃PS₄ (256 atoms) | N/A | 0.5 s | - |
| Li₃PS₄ (2000 atoms) | N/A | 5 s | - |

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@software{dft_lammps_framework,
  author = {Your Name},
  title = {DFT + LAMMPS Multi-Scale Materials Simulation Framework},
  year = {2026},
  url = {https://github.com/yourusername/dft-lammps-framework}
}
```

### Related Papers

- **DeePMD-kit**: Wang et al., Comput. Phys. Commun. 228, 178 (2018)
- **NEP**: Fan et al., Phys. Rev. B 104, 104309 (2021)
- **ASE**: Larsen et al., J. Phys. Condens. Matter 29, 273002 (2017)
- **Pymatgen**: Ong et al., Comput. Mater. Sci. 68, 314 (2013)

---

## 📞 Support

- 📧 Email: your.email@example.com
- 💬 Issues: [GitHub Issues](https://github.com/yourusername/dft-lammps-framework/issues)
- 📖 Documentation: [Full Documentation](https://your-docs-site.com)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

# 中文文档

## 概述

本框架整合了密度泛函理论(DFT)和分子动力学(MD)模拟，支持机器学习势函数(DeepMD、NEP)训练、主动学习和高通量材料筛选。

### 主要特性

- **DFT计算**: 支持VASP和Quantum ESPRESSO
- **ML势训练**: DeePMD-kit、NEP完整训练流程
- **主动学习**: 自动化探索-标注-重训练循环
- **高通量筛选**: 批量计算和工作流管理
- **HPC集成**: Slurm/PBS作业调度支持

### 快速开始

```bash
# 安装
conda create -n dft-lammps python=3.10
conda activate dft-lammps
pip install -r requirements.txt

# 运行示例
cd examples/quick_start
python simple_workflow.py
```

### 学习路径

1. **[快速入门](tutorials/01_quick_start.md)** - 15分钟上手
2. **[DFT基础](tutorials/02_dft_basics.md)** - VASP/QE计算设置
3. **[ML势训练](tutorials/03_ml_potential.md)** - 势函数训练
4. **[主动学习](tutorials/04_active_learning.md)** - 自动化优化
5. **[高通量筛选](tutorials/05_high_throughput.md)** - 批量计算
6. **[HPC部署](tutorials/06_hpc_deployment.md)** - 集群使用
7. **[高级工作流](tutorials/07_advanced_workflows.md)** - 定制开发

---

**Made with ❤️ for the computational materials science community**
