# DFT+LAMMPS集成工作流文档

## 目录
1. [概述](#概述)
2. [安装与依赖](#安装与依赖)
3. [快速开始](#快速开始)
4. [架构设计](#架构设计)
5. [模块详解](#模块详解)
6. [API参考](#api参考)
7. [配置指南](#配置指南)
8. [示例教程](#示例教程)
9. [故障排除](#故障排除)
10. [最佳实践](#最佳实践)

---

## 概述

### 什么是Integrated Materials Workflow?

`integrated_materials_workflow.py` 是一个统一的多尺度材料计算框架，整合了以下模块：

| 模块 | 功能 | 原始文件 |
|------|------|----------|
| **Structure Fetcher** | 从Materials Project或文件获取结构 | - |
| **DFT Bridge** | VASP/QE结构优化和AIMD | `dft_to_lammps_bridge.py` |
| **ML Training** | DeepMD/NEP机器学习势训练 | `nep_training_pipeline.py` + `active_learning_workflow.py` |
| **MD Simulation** | LAMMPS分子动力学 | `battery_screening_pipeline.py` |
| **Analysis** | 扩散分析、电导率预测 | - |

### 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    统一材料计算工作流                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Fetch Structure                                        │
│  ├── Materials Project API                                       │
│  └── 本地结构文件 (VASP/POSCAR, CIF, XYZ...)                      │
│                              ↓                                   │
│  Stage 2: DFT Calculation                                        │
│  ├── 结构优化 (VASP/Quantum ESPRESSO)                            │
│  └── AIMD训练数据生成                                            │
│                              ↓                                   │
│  Stage 3: ML Training                                            │
│  ├── 数据准备 (dpdata转换)                                       │
│  ├── DeepMD/NEP模型训练                                          │
│  └── 模型集成 (不确定性量化)                                     │
│                              ↓                                   │
│  Stage 4: MD Simulation                                          │
│  ├── LAMMPS输入生成                                              │
│  ├── 多温度MD模拟                                                │
│  └── 轨迹收集                                                    │
│                              ↓                                   │
│  Stage 5: Analysis                                               │
│  ├── 扩散系数计算 (MSD分析)                                      │
│  ├── 离子电导率 (Nernst-Einstein)                                │
│  └── Arrhenius拟合 (活化能)                                      │
│                              ↓                                   │
│  Output: 综合报告 + 性能预测                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 安装与依赖

### 系统要求

- Python 3.8+
- Linux/Unix环境 (推荐)
- 足够的存储空间 (>10GB用于DFT计算)

### Python依赖

```bash
pip install ase pymatgen mp-api dpdata numpy pandas
```

### DFT软件

#### VASP (推荐)
```bash
# 需要有效的VASP许可证
module load vasp/6.3.0
```

#### Quantum ESPRESSO
```bash
sudo apt-get install quantum-espresso
```

### ML势框架

#### DeepMD-kit
```bash
pip install deepmd-kit
```

#### GPUMD (NEP)
```bash
git clone https://github.com/brucefan1983/GPUMD.git
cd GPUMD
make -j4
```

### MD模拟器

```bash
# LAMMPS with ML potential support
# 需要安装USER-DEEPMD包
```

---

## 快速开始

### 命令行使用

#### 1. 从Materials Project ID运行

```bash
python integrated_materials_workflow.py \
    --mp-id mp-30406 \
    -o ./Li3PS4_output \
    --dft-code vasp \
    --ml-framework deepmd
```

#### 2. 从结构文件运行

```bash
python integrated_materials_workflow.py \
    --structure POSCAR \
    -o ./output \
    --formula "Li3PS4"
```

#### 3. 跳过某些阶段

```bash
# 已有DFT结果，跳过DFT和ML训练
python integrated_materials_workflow.py \
    --mp-id mp-30406 \
    --skip-dft \
    --skip-ml \
    -o ./output
```

### Python API使用

```python
from integrated_materials_workflow import (
    IntegratedMaterialsWorkflow,
    IntegratedWorkflowConfig,
    MaterialsProjectConfig,
    DFTStageConfig,
    MLPotentialConfig,
    MDStageConfig,
)

# 配置工作流
config = IntegratedWorkflowConfig(
    workflow_name="Li3PS4_Solid_Electrolyte",
    working_dir="./output",
    mp_config=MaterialsProjectConfig(api_key="your_api_key"),
    dft_config=DFTStageConfig(code="vasp", encut=520),
    ml_config=MLPotentialConfig(framework="deepmd", num_models=4),
    md_config=MDStageConfig(temperatures=[300, 500, 700, 900]),
)

# 创建并运行工作流
workflow = IntegratedMaterialsWorkflow(config)
results = workflow.run(material_id="mp-30406")

# 访问结果
print(f"DFT Energy: {results['dft']['energy_per_atom']} eV/atom")
print(f"Diffusion at 300K: {results['analysis']['diffusion_coefficients'][300]} cm²/s")
print(f"Activation Energy: {results['analysis']['activation_energy']} eV")
```

---

## 架构设计

### 类层次结构

```
IntegratedMaterialsWorkflow (主控制器)
    ├── ProgressMonitor (进度监控)
    ├── ErrorHandler (错误处理)
    ├── StructureFetcher (结构获取)
    ├── DFTStage (DFT计算)
    ├── MLTrainingStage (ML训练)
    ├── MDSimulationStage (MD模拟)
    └── AnalysisStage (分析)
```

### 数据流

```python
# 数据结构示意图
results = {
    'structure': Structure,  # Pymatgen Structure
    'formula': 'Li3PS4',
    'dft': {
        'energy': -123.45,
        'energy_per_atom': -4.12,
        'forces': [...],
        'stress': [...],
        'success': True
    },
    'ml_models': [
        './output/ml_models/model_0/graph.pb',
        './output/ml_models/model_1/graph.pb',
        ...
    ],
    'trajectories': {
        300: './output/md_results/T300/dump.lammpstrj',
        500: './output/md_results/T500/dump.lammpstrj',
        ...
    },
    'analysis': {
        'diffusion_coefficients': {300: 1.2e-7, 500: 5.6e-6, ...},
        'conductivities': {300: 2.3e-5, 500: 1.1e-3, ...},
        'activation_energy': 0.25,
        'pre_exponential': 1.5e-4
    }
}
```

### 阶段依赖关系

```
fetch_structure
     ↓
dft_calculation
     ↓
ml_training
     ↓
md_simulation
     ↓
   analysis
```

---

## 模块详解

### 1. StructureFetcher (结构获取)

**功能**: 从多种来源获取晶体结构

**支持来源**:
- Materials Project (通过API)
- 本地文件 (POSCAR, CIF, XYZ, etc.)

**示例代码**:

```python
from integrated_materials_workflow import StructureFetcher

fetcher = StructureFetcher(config, monitor, error_handler)

# 从MP获取
structures = fetcher.fetch_from_mp(material_id="mp-30406")
structures = fetcher.fetch_from_mp(formula="Li3PS4")
structures = fetcher.fetch_from_mp(chemsys="Li-P-S")

# 从文件获取
structures = fetcher.fetch_from_file("./POSCAR")
```

### 2. DFTStage (DFT计算)

**功能**: 执行第一性原理计算

**支持代码**: VASP, Quantum ESPRESSO

**计算类型**:
- 结构优化 (ISIF=3)
- 单点能计算
- AIMD训练数据生成

**示例代码**:

```python
from integrated_materials_workflow import DFTStage

dft = DFTStage(config, monitor, error_handler)

# 结构优化
results = dft.run_relaxation(
    structure=structure,
    output_dir="./dft_results"
)

# AIMD生成训练数据
dft.run_aimd(
    structure=structure,
    temperature=900,
    nsteps=10000,
    output_dir="./aimd_results"
)
```

### 3. MLTrainingStage (ML势训练)

**功能**: 训练机器学习势函数

**支持框架**: DeepMD-kit, NEP (GPUMD)

**特性**:
- 自动数据准备 (VASP→DeepMD格式)
- 模型集成训练 (用于不确定性量化)
- 训练过程监控

**示例代码**:

```python
from integrated_materials_workflow import MLTrainingStage

ml = MLTrainingStage(config, monitor, error_handler)

# 准备训练数据
train_dir, valid_dir = ml.prepare_data(
    dft_output_dirs=["./dft_results"],
    output_dir="./training_data"
)

# 训练DeepMD模型
model_paths = ml.train_deepmd(
    train_dir=train_dir,
    valid_dir=valid_dir,
    type_map=["Li", "P", "S"],
    output_dir="./ml_models"
)

# 训练NEP模型
nep_model = ml.train_nep(
    train_xyz="train.xyz",
    test_xyz="test.xyz",
    type_list=["Li", "P", "S"],
    output_dir="./nep_models"
)
```

### 4. MDSimulationStage (MD模拟)

**功能**: 大规模分子动力学模拟

**支持引擎**: LAMMPS

**支持势函数**: DeepMD, NEP, SNAP, etc.

**示例代码**:

```python
from integrated_materials_workflow import MDSimulationStage

md = MDSimulationStage(config, monitor, error_handler)

# 单温度MD
traj_file = md.run_lammps_md(
    structure=structure,
    model_path="graph.pb",
    temperature=500,
    output_dir="./md_T500"
)

# 多温度MD
trajectories = md.run_multi_temperature(
    structure=structure,
    model_path="graph.pb",
    output_base_dir="./md_results"
)
```

### 5. AnalysisStage (分析)

**功能**: 分析MD轨迹，预测材料性能

**分析类型**:
- 扩散系数 (MSD分析)
- 离子电导率 (Nernst-Einstein方程)
- 活化能 (Arrhenius拟合)

**示例代码**:

```python
from integrated_materials_workflow import AnalysisStage

analysis = AnalysisStage(config, monitor, error_handler)

# 扩散系数
D = analysis.analyze_diffusion(
    trajectory_file="dump.lammpstrj",
    atom_type="Li",
    timestep=1.0
)

# 离子电导率
sigma = analysis.compute_conductivity(
    D=1e-6,
    structure=structure,
    temperature=300,
    ion_type="Li"
)

# Arrhenius拟合
Ea, D0 = analysis.fit_arrhenius(
    temperatures=[300, 400, 500, 600, 700],
    diffusion_coeffs=[1e-7, 5e-7, 2e-6, 6e-6, 1.5e-5]
)

# 完整分析
results = analysis.run_full_analysis(
    trajectories={300: "traj_300", 500: "traj_500", ...},
    structure=structure,
    output_dir="./analysis"
)
```

---

## API参考

### 配置类

#### IntegratedWorkflowConfig

```python
@dataclass
class IntegratedWorkflowConfig:
    workflow_name: str              # 工作流名称
    working_dir: str                # 工作目录
    mp_config: MaterialsProjectConfig      # MP配置
    dft_config: DFTStageConfig             # DFT配置
    ml_config: MLPotentialConfig           # ML配置
    md_config: MDStageConfig               # MD配置
    analysis_config: AnalysisConfig        # 分析配置
    stages: Dict[str, WorkflowStage]       # 阶段控制
    max_parallel: int               # 最大并行数
    save_intermediate: bool         # 保存中间结果
    generate_report: bool           # 生成报告
```

#### DFTStageConfig

```python
@dataclass
class DFTStageConfig:
    code: str = "vasp"              # vasp, espresso
    functional: str = "PBE"         # 泛函
    encut: float = 520              # 截断能 (eV)
    kpoints_density: float = 0.25   # k点密度
    ediff: float = 1e-6             # 电子收敛
    ncores: int = 32                # 并行核数
    max_steps: int = 200            # 最大离子步
    fmax: float = 0.01              # 力收敛标准
```

#### MLPotentialConfig

```python
@dataclass
class MLPotentialConfig:
    framework: str = "deepmd"       # deepmd, nep, mace
    preset: str = "fast"            # fast, accurate, light
    num_models: int = 4             # 集成模型数
    max_iterations: int = 10        # 主动学习迭代
    uncertainty_threshold: float = 0.15  # 不确定性阈值
```

#### MDStageConfig

```python
@dataclass
class MDStageConfig:
    ensemble: str = "nvt"           # nve, nvt, npt
    temperatures: List[float]       # 温度列表
    timestep: float = 1.0           # 时间步长 (fs)
    nsteps_equil: int = 50000       # 平衡步数
    nsteps_prod: int = 500000       # 生产步数
    nprocs: int = 4                 # LAMMPS并行数
```

### 主工作流类

#### IntegratedMaterialsWorkflow

```python
class IntegratedMaterialsWorkflow:
    def __init__(self, config: IntegratedWorkflowConfig)
    
    def run(self, 
            material_id: Optional[str] = None,
            structure_file: Optional[str] = None,
            formula: Optional[str] = None) -> Dict
    """
    运行完整工作流
    
    Args:
        material_id: Materials Project ID (e.g., "mp-30406")
        structure_file: 本地结构文件路径
        formula: 化学式 (用于搜索)
    
    Returns:
        包含所有结果的字典
    """
```

---

## 配置指南

### 配置文件示例

```python
# config.py
from integrated_materials_workflow import *

# 高性能计算配置
hpc_config = IntegratedWorkflowConfig(
    workflow_name="HPC_Calculation",
    working_dir="./hpc_output",
    
    dft_config=DFTStageConfig(
        code="vasp",
        encut=600,
        ncores=64,
        fmax=0.005,  # 严格收敛
    ),
    
    ml_config=MLPotentialConfig(
        framework="deepmd",
        preset="accurate",
        num_models=4,
    ),
    
    md_config=MDStageConfig(
        temperatures=[300, 400, 500, 600, 700, 800, 900, 1000],
        nsteps_equil=200000,   # 200 ps平衡
        nsteps_prod=1000000,   # 1 ns生产
        nprocs=8,
    ),
)

# 快速测试配置
quick_config = IntegratedWorkflowConfig(
    workflow_name="Quick_Test",
    working_dir="./quick_output",
    
    dft_config=DFTStageConfig(
        encut=400,  # 降低精度
        fmax=0.05,  # 宽松收敛
        max_steps=50,
    ),
    
    ml_config=MLPotentialConfig(
        preset="fast",
        num_models=1,  # 单模型
    ),
    
    md_config=MDStageConfig(
        temperatures=[300, 500, 700],
        nsteps_equil=10000,   # 10 ps
        nsteps_prod=50000,    # 50 ps
    ),
)
```

### 环境变量

```bash
# Materials Project API Key
export MP_API_KEY="your_api_key_here"

# VASP设置
export VASP_PP_PATH="/path/to/pseudopotentials"
export VASP_COMMAND="mpirun -np 32 vasp_std"

# DeepMD设置
export DP_PATH="/path/to/deepmd"

# LAMMPS设置
export LAMMPS_COMMAND="lmp"
```

---

## 示例教程

### 教程1: 固态电解质筛选 (Li3PS4)

```python
#!/usr/bin/env python3
"""
Li3PS4离子电导率预测完整流程
"""

from integrated_materials_workflow import *

# 1. 配置工作流
config = IntegratedWorkflowConfig(
    workflow_name="Li3PS4_Conductivity",
    working_dir="./Li3PS4_output",
    
    dft_config=DFTStageConfig(code="vasp", encut=520),
    ml_config=MLPotentialConfig(framework="deepmd", num_models=4),
    md_config=MDStageConfig(
        temperatures=[300, 400, 500, 600, 700, 800, 900],
        nsteps_prod=500000,
    ),
)

# 2. 运行工作流
workflow = IntegratedMaterialsWorkflow(config)
results = workflow.run(material_id="mp-30406")

# 3. 分析结果
analysis = results['analysis']

print("="*60)
print("Li3PS4离子传输性能")
print("="*60)
print(f"\n扩散系数 (cm²/s):")
for T, D in sorted(analysis['diffusion_coefficients'].items()):
    print(f"  {T}K: {D:.2e}")

print(f"\n离子电导率 (S/cm):")
for T, sigma in sorted(analysis['conductivities'].items()):
    print(f"  {T}K: {sigma:.2e}")

print(f"\n活化能: {analysis['activation_energy']:.3f} eV")
print(f"\n室温电导率: {analysis['conductivities'][300]:.2e} S/cm")
```

### 教程2: 批量材料筛选

```python
#!/usr/bin/env python3
"""
批量筛选多种材料
"""

from integrated_materials_workflow import *
import pandas as pd

# 候选材料列表
candidates = [
    "mp-30406",   # Li3PS4
    "mp-28750",   # Li7P3S11
    "mp-696138",  # LGPS
]

results_list = []

for mp_id in candidates:
    print(f"\nProcessing {mp_id}...")
    
    config = IntegratedWorkflowConfig(
        workflow_name=f"screening_{mp_id}",
        working_dir=f"./screening/{mp_id}",
    )
    
    workflow = IntegratedMaterialsWorkflow(config)
    
    try:
        results = workflow.run(material_id=mp_id)
        
        results_list.append({
            'material_id': mp_id,
            'formula': results['formula'],
            'energy_per_atom': results['dft']['energy_per_atom'],
            'conductivity_300K': results['analysis']['conductivities'][300],
            'activation_energy': results['analysis']['activation_energy'],
        })
    except Exception as e:
        print(f"Failed: {e}")

# 生成排名
df = pd.DataFrame(results_list)
df = df.sort_values('conductivity_300K', ascending=False)
print("\n筛选结果排名:")
print(df)
```

### 教程3: 自定义分析

```python
#!/usr/bin/env python3
"""
自定义后处理分析
"""

from integrated_materials_workflow import AnalysisStage
import numpy as np
import matplotlib.pyplot as plt

# 加载已有轨迹
analysis = AnalysisStage(config=None, monitor=None, error_handler=None)

# 计算径向分布函数 (RDF)
def compute_rdf(trajectory_file, atom_type1, atom_type2):
    """计算径向分布函数"""
    from ase.io import read
    from ase.geometry.analysis import Analysis
    
    frames = read(trajectory_file, index=':', format='lammps-dump-text')
    
    rdf_data = []
    for frame in frames[::10]:  # 每10帧采样
        ana = Analysis(frame)
        rdf = ana.get_rdf(rmax=10.0, nbins=100, 
                         elements=[atom_type1, atom_type2])
        rdf_data.append(rdf)
    
    return np.mean(rdf_data, axis=0)

# 使用示例
rdf = compute_rdf("dump.lammpstrj", "Li", "S")

plt.plot(rdf)
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.title("Li-S RDF")
plt.savefig("rdf.png")
```

---

## 故障排除

### 常见问题

#### 1. Materials Project API错误

**问题**: `MPRestError: API key not found`

**解决方案**:
```bash
export MP_API_KEY="your_api_key"
# 或在代码中设置
config.mp_config.api_key = "your_api_key"
```

#### 2. VASP计算失败

**问题**: `VASP calculation failed with return code 1`

**检查清单**:
- [ ] 检查VASP是否安装: `which vasp_std`
- [ ] 检查伪势路径: `echo $VASP_PP_PATH`
- [ ] 检查INCAR参数是否合理
- [ ] 查看vasp.log错误信息

#### 3. DeepMD训练失败

**问题**: `No valid data found for training`

**解决方案**:
```python
# 确保DFT输出目录包含有效的OUTCAR
# 手动检查
dft_dir = "./dft_results"
import os
assert os.path.exists(f"{dft_dir}/OUTCAR"), "OUTCAR not found"
```

#### 4. LAMMPS模拟失败

**问题**: `LAMMPS simulation failed`

**检查清单**:
- [ ] LAMMPS是否安装: `which lmp`
- [ ] 是否安装了USER-DEEPMD包: `lmp -h | grep deepmd`
- [ ] 模型文件路径是否正确
- [ ] 检查structure.data格式

#### 5. 扩散系数为0

**问题**: 计算的扩散系数为0或负数

**可能原因**:
- MD时间太短，扩散距离不足
- 温度太低，未发生扩散
- MSD计算起始时间设置不当

**解决方案**:
```python
# 增加MD模拟时间
config.md_config.nsteps_prod = 1000000  # 1 ns

# 提高温度
config.md_config.temperatures = [500, 700, 900, 1100]
```

---

## 最佳实践

### 1. 计算效率优化

```python
# 使用较小的超胞进行快速测试
config.dft_config.max_steps = 50  # 快速收敛测试

# 使用较粗的k点网格
config.dft_config.kpoints_density = 0.3

# 减少MD步数
config.md_config.nsteps_prod = 100000
```

### 2. 精度提升

```python
# 高精度DFT
config.dft_config.encut = 600
config.dft_config.fmax = 0.005

# 更多ML模型
config.ml_config.num_models = 8

# 更长MD模拟
config.md_config.nsteps_prod = 2000000  # 2 ns
```

### 3. 批量任务管理

```python
# 使用工作队列管理多个任务
from concurrent.futures import ProcessPoolExecutor

def run_single_material(mp_id):
    config = IntegratedWorkflowConfig(
        workflow_name=mp_id,
        working_dir=f"./batch/{mp_id}",
    )
    workflow = IntegratedMaterialsWorkflow(config)
    return workflow.run(material_id=mp_id)

# 并行运行
materials = ["mp-30406", "mp-28750", "mp-696138"]
with ProcessPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(run_single_material, materials))
```

### 4. 结果验证

```python
# 检查DFT收敛
def validate_dft_results(results):
    dft = results.get('dft', {})
    assert dft.get('success'), "DFT calculation failed"
    assert dft['energy_per_atom'] < 0, "Positive energy per atom"
    max_force = np.max(np.abs(dft['forces']))
    assert max_force < 0.05, f"Large max force: {max_force}"
    return True

# 检查ML模型
def validate_ml_models(results):
    models = results.get('ml_models', [])
    assert len(models) >= 1, "No ML models trained"
    for model in models:
        assert os.path.exists(model), f"Model not found: {model}"
    return True
```

### 5. 资源管理

```python
# 监控资源使用
import psutil
import time

def monitor_resources():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"CPU: {cpu_percent}%, Memory: {memory.percent}%")

# 定期监控
while workflow_running:
    monitor_resources()
    time.sleep(60)
```

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0.0 | 2026-03-09 | 初始版本，整合DFT+LAMMPS+ML工作流 |

## 引用

如果使用本工作流，请引用:

```bibtex
@software{integrated_materials_workflow,
  title = {Integrated Materials Workflow: DFT+LAMMPS+ML Pipeline},
  author = {DFT-MD Coupling Expert},
  year = {2026},
  url = {https://github.com/your-repo}
}
```

## 联系方式

- 问题报告: [GitHub Issues](https://github.com/your-repo/issues)
- 文档: [ReadTheDocs](https://your-docs.readthedocs.io)
- 邮箱: your.email@example.com

---

*文档版本: 1.0.0 | 最后更新: 2026-03-09*
