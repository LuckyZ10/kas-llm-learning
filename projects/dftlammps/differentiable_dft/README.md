# Differentiable DFT Module

可微分密度泛函理论(DFT)模块 - 基于自动微分的量子力学计算与材料逆向设计。

## 模块结构

```
differentiable_dft/
├── jax_dft_interface.py          # JAX-DFT接口 (741行)
├── dftk_julia_interface.py       # DFTK.jl Julia接口 (660行)
├── enhanced_autodiff.py          # 增强自动微分功能 (500+行)
├── inverse_design/               # 逆向设计模块
│   ├── __init__.py               # 模块初始化
│   ├── core.py                   # 核心逆向设计框架 (500+行)
│   ├── band_gap_design.py        # 带隙逆向设计 (500+行)
│   └── ion_conductor_design.py   # 离子导体逆向设计 (550+行)
├── case_inverse_bandgap/         # 带隙逆向设计案例
│   └── main.py                   # 案例主程序 (400+行)
└── case_inverse_ion_conductor/   # 离子导体逆向设计案例
    └── main.py                   # 案例主程序 (400+行)
```

## 核心功能

### 1. JAX-DFT接口 (`jax_dft_interface.py`)

基于JAX的端到端可微分DFT实现：

- **核心类:**
  - `DifferentiableDFT`: 可微分DFT计算器
  - `JAXGrid`: 实空间网格管理
  - `LDAFunctional`: LDA交换关联泛函
  - `NeuralDFT`: 神经网络混合DFT

- **功能特性:**
  - 能量自动微分
  - 原子力解析梯度 (Hellmann-Feynman + Pulay)
  - 应力张量自动微分
  - 自洽场(SCF)迭代
  - HGH赝势支持

### 2. DFTK.jl接口 (`dftk_julia_interface.py`)

Julia DFTK.jl的Python封装：

- **核心类:**
  - `DFTKInterface`: DFTK.jl主接口
  - `DFTKAutodiff`: 自动微分接口
  - `GeometryGradientFlow`: 几何优化梯度流
  - `BandStructureCalculator`: 能带结构计算

- **功能特性:**
  - 平面波DFT计算
  - Zygote/ForwardDiff自动微分
  - 力/应力计算
  - 带隙计算
  - 结构优化

### 3. 增强自动微分 (`enhanced_autodiff.py`)

扩展的可微分功能：

- **高阶导数:**
  - `HighOrderDerivatives`: 二阶、三阶导数
  - 力常数矩阵
  - 声子频率
  - 弹性常数张量

- **响应性质:**
  - `ResponseProperties`: 极化率、介电常数
  - `TDDFTCalculator`: 线性响应TDDFT
  - Born有效电荷

- **超参数优化:**
  - 混合参数自动优化
  - 截断能优化

## 逆向设计模块

### 核心框架 (`inverse_design/core.py`)

逆向设计的基础架构：

```python
from inverse_design import (
    DesignTarget,
    FractionalCoordinateStructure,
    InverseDesignOptimizer
)

# 定义目标
target = DesignTarget(
    target_type='band_gap',
    target_value=1.5,
    tolerance=0.05
)

# 创建参数化结构
structure = FractionalCoordinateStructure(
    n_atoms=4,
    atomic_numbers=jnp.array([14, 14, 8, 8]),
    initial_cell=jnp.eye(3) * 10.0
)

# 优化
optimizer = InverseDesignOptimizer(objective)
result = optimizer.optimize(structure)
```

### 带隙逆向设计 (`inverse_design/band_gap_design.py`)

设计具有目标带隙的材料：

- **BandGapCalculator**: 带隙计算
- **SolarCellOptimizer**: 太阳能电池材料优化
- **LEDMaterialDesigner**: LED材料设计 (RGB)
- **TransparentConductorOptimizer**: 透明导电氧化物

### 离子导体逆向设计 (`inverse_design/ion_conductor_design.py`)

设计高离子电导率材料：

- **IonMigrationAnalyzer**: 离子迁移分析
- **SolidElectrolyteDesigner**: 固态电解质设计
- **NASICONDesigner**: NASICON结构优化
- **SulfideElectrolyteDesigner**: 硫化物电解质设计

## 应用案例

### 案例1: 带隙逆向设计 (`case_inverse_bandgap/`)

运行: `python case_inverse_bandgap/main.py`

包含三个子案例：
1. **太阳能电池材料设计**: 目标带隙 ~1.3 eV
2. **LED材料设计**: 红、绿、蓝三色LED
3. **透明导电氧化物**: 宽带隙 + 高透明

### 案例2: 离子导体逆向设计 (`case_inverse_ion_conductor/`)

运行: `python case_inverse_ion_conductor/main.py`

包含三个子案例：
1. **锂离子导体**: 硫化物电解质设计
2. **NASICON优化**: Si/P组成优化
3. **硫化物超离子导体**: 类LGPS材料

## 使用示例

### 基础DFT计算

```python
from jax_dft_interface import DFTConfig, SystemConfig, DifferentiableDFT

# 配置DFT
config = DFTConfig(
    xc_functional='lda_x+lda_c_pw',
    grid_spacing=0.16,
    ecut=30.0
)

# 创建计算器
dft = DifferentiableDFT(config)

# 定义系统
system = SystemConfig(
    positions=jnp.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]]),
    atomic_numbers=jnp.array([14, 14]),
    cell=jnp.eye(3) * 10.0
)

# 计算能量和力
result = dft.compute_energy(system, return_forces=True)
print(f"Energy: {result['energy']} Ha")
print(f"Forces:\n{result['forces']}")
```

### 带隙逆向设计

```python
from inverse_design import (
    BandGapTarget, BandGapCalculator,
    FractionalCoordinateStructure, InverseDesignOptimizer
)

# 创建带隙计算器
calc = BandGapCalculator(dft_engine)

# 定义目标
target = BandGapTarget(
    target_gap=1.5,  # eV
    gap_type='direct',
    tolerance=0.05
)

# 目标函数
objective = BandGapObjective(calc, target)

# 优化
structure = FractionalCoordinateStructure(...)
optimizer = InverseDesignOptimizer(
    lambda p, s: objective.loss(*s.to_structure()),
    learning_rate=0.01
)
result = optimizer.optimize(structure)
```

### 离子导体设计

```python
from inverse_design import (
    IonConductorTarget, SolidElectrolyteDesigner
)

# 设计目标
target = IonConductorTarget(
    ion_type='Li',
    target_conductivity=1e-3,  # S/cm
    min_migration_barrier=0.25  # eV
)

# 设计器
designer = SolidElectrolyteDesigner(dft_engine)
result = designer.design_lithium_conductor(initial_structure)
```

## 依赖项

### Python依赖
```
jax>=0.4.0
jaxlib>=0.4.0
numpy>=1.20.0
matplotlib>=3.5.0
```

### Julia依赖 (可选)
```julia
using Pkg
Pkg.add("DFTK")
Pkg.add("Zygote")
Pkg.add("ForwardDiff")
```

## 安装与运行

### 安装

```bash
# 克隆仓库
cd dftlammps/differentiable_dft

# 安装Python依赖
pip install jax jaxlib numpy matplotlib

# 安装Julia依赖 (可选)
julia -e 'using Pkg; Pkg.add("DFTK")'
```

### 运行示例

```bash
# 运行JAX-DFT示例
python jax_dft_interface.py

# 运行DFTK接口示例
python dftk_julia_interface.py

# 运行增强自动微分示例
python enhanced_autodiff.py

# 运行逆向设计示例
python -m inverse_design.core
python -m inverse_design.band_gap_design
python -m inverse_design.ion_conductor_design

# 运行完整案例
python case_inverse_bandgap/main.py
python case_inverse_ion_conductor/main.py
```

## 性能优化

### JAX优化
- 使用`@jit`编译关键函数
- 使用`vmap`向量化批量计算
- 启用XLA GPU加速

### DFTK优化
- 多线程并行: `ENV["JULIA_NUM_THREADS"] = 4`
- 选择合适的对角化算法
- 使用混合加速收敛

## 参考文献

1. Bradbury, J., et al. (2018). JAX: composable transformations of Python+NumPy programs.
2. Herbst, M. F., et al. (2021). DFTK: A Julian approach for simulating electrons in solids.
3. Innes, M., et al. (2019). A differentiable programming system to bridge machine learning and scientific computing.
4. Schleder, G. R., et al. (2019). From DFT to machine learning: recent approaches to materials science.

## 许可证

MIT License
