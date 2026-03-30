# 先进机器学习势能技术文档

## 概述

本模块提供统一的接口访问多种先进的机器学习势能模型，包括：

- **MACE**: 高阶等变消息传递势
- **CHGNet**: 晶体哈密顿图神经网络势
- **Orb**: 轨道表示超快推理势

## 快速开始

```python
from dftlammps.advanced_potentials import load_ml_potential

# 加载MACE
mace_calc = load_ml_potential("mace", model_name="medium")

# 加载CHGNet
chgnet_calc = load_ml_potential("chgnet")

# 加载Orb
orb_calc = load_ml_potential("orb", device="cuda")
```

## 1. MACE接口

### 1.1 基本使用

```python
from dftlammps.advanced_potentials import MACEWorkflow, MACEConfig

# 配置
config = MACEConfig(
    model_type="medium",  # small, medium, large
    device="cuda",
    cutoff=6.0
)

# 创建工作流
workflow = MACEWorkflow(config)
workflow.setup_calculator()

# 预测单个结构
from ase.io import read
atoms = read("structure.xyz")
result = workflow.predict_structure(atoms)

print(f"Energy: {result['energy']:.4f} eV")
print(f"Max force: {np.max(np.abs(result['forces'])):.4f} eV/Å")
```

### 1.2 MD模拟

```python
from dftlammps.advanced_potentials import MACEMDConfig

md_config = MACEMDConfig(
    temperature=300.0,
    n_steps=10000,
    timestep=1.0,
    trajectory_file="mace_md.traj"
)

trajectory = workflow.run_md_simulation("initial.xyz", md_config)
```

### 1.3 结构优化

```python
relaxed = workflow.relax_structure(
    atoms,
    fmax=0.01,  # eV/Å
    optimizer="LBFGS"
)
```

### 1.4 主动学习

```python
from dftlammps.advanced_potentials import MACEActiveLearningConfig, MACEActiveLearning

al_config = MACEActiveLearningConfig(
    force_uncertainty_threshold=0.1,
    max_iterations=10
)

active_learning = MACEActiveLearning(calculator, al_config)
new_training_data = active_learning.run(initial_structure)
```

## 2. CHGNet接口

### 2.1 基本使用

```python
from dftlammps.advanced_potentials import CHGNetWorkflow, CHGNetConfig

config = CHGNetConfig(
    model_name="0.3.0",
    predict_magmom=True,
    use_device="cuda"
)

workflow = CHGNetWorkflow(config)
workflow.setup()

# 预测（包含磁矩）
atoms = read("magnetic_structure.cif")
prediction = workflow.predict_structure(atoms)

# 获取磁矩信息
mag_props = workflow.calculator.get_magmom_prediction(atoms)
print(f"Total magnetic moment: {mag_props.total_magmom:.2f} μB")
print(f"Magnetic ordering: {mag_props.magnetic_ordering}")
```

### 2.2 批量筛选磁性材料

```python
structures = [read(f) for f in structure_files]
pipeline = CHGNetScreeningPipeline(workflow.calculator)

magnetic_materials, results_df = pipeline.predictor.screen_magnetic_materials(
    structures,
    min_magmom=0.5  # μB per atom
)
```

### 2.3 EOS计算

```python
from dftlammps.advanced_potentials import CHGNetStructureOptimizer

optimizer = CHGNetStructureOptimizer(workflow.calculator)
eos_data = optimizer.calculate_eos(atoms, n_points=7)

print(f"Equilibrium volume: {eos_data['V0']:.2f} Å³")
print(f"Bulk modulus: {eos_data['B']/1e9:.2f} GPa")
```

### 2.4 DFT预筛选

```python
from dftlammps.advanced_potentials import CHGNetDFTInterface

dft_interface = CHGNetDFTInterface(workflow.calculator)

# 预筛选候选结构
selected = dft_interface.prescreen_for_dft(
    candidate_structures,
    n_select=10,
    criteria="energy"  # or "diversity"
)

# 使用CHGNet初始化DFT计算
initialized = dft_interface.run_dft_with_initialization(
    selected,
    dft_calculator=vasp_calc
)
```

## 3. Orb接口

### 3.1 基本使用

```python
from dftlammps.advanced_potentials import OrbWorkflow, OrbConfig

config = OrbConfig(
    model_name="orb_v2",
    device="cuda",
    precision="float32",
    batch_size=64
)

workflow = OrbWorkflow(config)
workflow.setup()

# 快速预测
result = workflow.predict_single(atoms)
```

### 3.2 批量预测（高性能）

```python
structures = [read(f) for f in structure_files]

# 批量预测
df = workflow.batch_predict(structures)
print(f"Throughput: {len(structures)/df['inference_time'].sum():.1f} structures/s")

# 性能基准测试
benchmark = workflow.benchmark(structures)
print(f"Atoms/s: {benchmark.atoms_per_second:.1f}")
```

### 3.3 高通量筛选

```python
from dftlammps.advanced_potentials import OrbScreeningPipeline

pipeline = OrbScreeningPipeline(workflow.calculator, batch_size=64)

# 按能量筛选
selected, results = pipeline.screen_by_energy(structures, top_k=100)

# 多样性聚类选择
diverse_selection = pipeline.cluster_and_select(
    structures,
    n_clusters=10,
    n_per_cluster=5
)
```

### 3.4 大规模MD

```python
from dftlammps.advanced_potentials import OrbMDConfig

md_config = OrbMDConfig(
    temperature=300.0,
    n_steps=1000000,  # 百万步长模拟
    output_interval=1000
)

simulator = workflow.run_md("initial.xyz", md_config)

# 计算RDF
r, rdf = simulator.compute_rdf(r_max=10.0, nbins=100)
```

## 4. 统一接口

### 4.1 势选择器

```python
from dftlammps.advanced_potentials import MLPotentialSelector

# 自动推荐
recommended = MLPotentialSelector.recommend(
    use_case="magnetic materials",
    priority_capabilities=["magmom"],
    speed_preference="balanced"
)

# 比较不同势
comparison = MLPotentialSelector.compare_potentials()
```

### 4.2 统一计算器

```python
from dftlammps.advanced_potentials import (
    UnifiedMLPotentialCalculator,
    UnifiedMLPotentialConfig,
    MLPotentialType
)

config = UnifiedMLPotentialConfig(
    potential_type=MLPotentialType.MACE,
    device="cuda"
)

calculator = UnifiedMLPotentialCalculator(config)

# 检查能力
if calculator.supports(MLPotentialCapability.MAGMOM):
    print("Supports magnetic moment prediction")

# 通用计算
calculator.calculate(atoms, properties=['energy', 'forces', 'stress'])
```

### 4.3 统一工作流

```python
from dftlammps.advanced_potentials import UnifiedMLPotentialWorkflow

workflow = UnifiedMLPotentialWorkflow(config)

# 预测
result = workflow.predict(atoms)

# 弛豫
relaxed = workflow.relax_structure(atoms, fmax=0.01)

# MD
trajectory = workflow.run_md(initial_structure, temperature=300, n_steps=10000)
```

## 5. 模型能力对比

| 特性 | MACE | CHGNet | Orb |
|------|------|--------|-----|
| 能量预测 | ✅ | ✅ | ✅ |
| 力预测 | ✅ | ✅ | ✅ |
| 应力预测 | ✅ | ✅ | ✅ |
| 磁矩预测 | ❌ | ✅ | ❌ |
| 电荷预测 | ❌ | ✅ | ❌ |
| 推理速度 | 中等 | 快 | 超快 |
| 训练数据 | MP+ | MP | MP+OMAT |
| 精度 | 最高 | 高 | 高 |
| 推荐场景 | 高精度MD | 磁性材料 | 大规模筛选 |

## 6. 性能优化

### 6.1 GPU加速

```python
# 所有势都支持CUDA加速
config = MACEConfig(device="cuda")
config = CHGNetConfig(use_device="cuda")
config = OrbConfig(device="cuda")

# 多GPU
config = MACEConfig(device="cuda:1")
```

### 6.2 批处理优化

```python
# 增大批大小提高吞吐量
orb_config = OrbConfig(batch_size=128)
chgnet_config = CHGNetConfig(batch_size=32)
```

### 6.3 精度设置

```python
# Orb支持多种精度
orb_config = OrbConfig(precision="float16")  # 最快
orb_config = OrbConfig(precision="float32")  # 平衡
orb_config = OrbConfig(precision="float64")  # 最高精度
```

## 7. 与LAMMPS集成

### 7.1 MACE-LAMMPS

```python
from dftlammps.advanced_potentials import MACELAMMPSInterface

mace_lammps = MACELAMMPSInterface("mace_model.pt")
mace_lammps.export_to_lammps_format("./lammps_potential")
mace_lammps.generate_lammps_input(
    structure_file="structure.data",
    elements=["Li", "P", "S"],
    output_file="in.lammps"
)
```

### 7.2 DeePMD/NEP兼容

DeePMD和NEP接口继承自核心模块：

```python
from dftlammps.core.ml_potential import NEPTrainingPipeline

nep_pipeline = NEPTrainingPipeline()
```

## 8. 训练数据准备

### 8.1 MACE数据集

```python
from dftlammps.advanced_potentials import MACEDatasetPreparer

preparer = MACEDatasetPreparer(cutoff=6.0)
preparer.load_vasp_data(["OUTCAR1", "OUTCAR2"])
preparer.load_extxyz("trajectory.xyz")

# 过滤
preparer.filter_frames(
    energy_threshold=50.0,
    force_threshold=50.0
)

# 准备数据集
dataset = preparer.prepare_dataset(
    train_ratio=0.9,
    val_ratio=0.05,
    test_ratio=0.05
)

# 统计信息
stats = preparer.get_statistics()
```

## 9. 命令行工具

### 9.1 MACE

```bash
# 预测
python -m dftlammps.advanced_potentials.mace_interface predict structure.xyz

# MD模拟
python -m dftlammps.advanced_potentials.mace_interface md structure.xyz \
    --temperature 300 --steps 10000

# 结构弛豫
python -m dftlammps.advanced_potentials.mace_interface relax structure.xyz \
    --fmax 0.01
```

### 9.2 CHGNet

```bash
# 批量预测
python -m dftlammps.advanced_potentials.chgnet_interface batch *.cif \
    --output results.csv

# 筛选磁性材料
python -m dftlammps.advanced_potentials.chgnet_interface screen *.cif \
    --magnetic
```

### 9.3 Orb

```bash
# 性能基准测试
python -m dftlammps.advanced_potentials.orb_interface benchmark *.xyz

# MD模拟
python -m dftlammps.advanced_potentials.orb_interface md structure.xyz \
    --temperature 500 --steps 100000
```

## 10. 最佳实践

### 10.1 选择合适的势

- **高精度MD模拟**: MACE (medium/large)
- **磁性材料研究**: CHGNet
- **大规模筛选/高通量**: Orb
- **实时模拟/在线学习**: Orb

### 10.2 收敛性检查

```python
# 检查力收敛
forces = atoms.get_forces()
max_force = np.max(np.abs(forces))
assert max_force < 0.01  # eV/Å

# 检查能量收敛
energy = atoms.get_potential_energy()
energy_per_atom = energy / len(atoms)
```

### 10.3 不确定性量化

```python
# 主动学习中检查不确定性
uncertainty = np.std(forces)
if uncertainty > threshold:
    # 需要DFT标记
    label_with_dft(atoms)
```

## 11. 参考文献

1. Batatia, I., et al. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. arXiv:2206.07697.

2. Deng, B., et al. (2023). CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling. Nature Machine Intelligence, 5(9), 1031-1041.

3. Stärk, W., et al. (2024). Orb: A fast, scalable neural network potential. arXiv:2410.22581.

4. Zhang, Y., et al. (2018). Deep potential molecular dynamics: a scalable model with the accuracy of quantum mechanics. Physical Review Materials, 2(2), 023803.

5. Fan, Z., et al. (2022). Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport. Physical Review B, 104(10), 104309.
