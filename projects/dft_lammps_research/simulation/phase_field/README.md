"""
Phase Field - DFT Multi-scale Coupling Module
=============================================
相场-DFT多尺度耦合模块

填补微观(DFT/MD)到介观(相场)的尺度鸿沟，实现跨尺度材料模拟。

核心特性
--------
1. **物理模型**
   - Cahn-Hilliard方程 (成分场演化)
   - Allen-Cahn方程 (序参量演化)
   - 电化学相场 (离子传输+反应动力学)
   - 力学-化学耦合 (应力场影响扩散)

2. **多尺度耦合**
   - 从DFT获取热力学参数 (化学势、界面能)
   - 从MD获取动力学参数 (扩散系数、迁移势垒)
   - 相场结果反馈给DFT (界面结构、缺陷构型)
   - 自动化参数传递

3. **数值实现**
   - 有限差分/有限元求解器
   - GPU加速 (CuPy/Numba)
   - 并行域分解
   - 自适应网格细化

4. **应用场景**
   - 锂离子电池SEI生长模拟
   - 合金沉淀相演化
   - 固态电解质晶界迁移
   - 催化剂表面重构

快速开始
--------

### 安装依赖

```bash
pip install numpy scipy matplotlib
pip install cupy  # 可选，用于GPU加速
pip install ase pymatgen  # 可选，用于DFT/MD接口
```

### 基本使用

```python
from phase_field.core.cahn_hilliard import CahnHilliardSolver, CahnHilliardConfig

# 配置
config = CahnHilliardConfig(
    nx=128, ny=128,
    dx=1.0, dt=0.001,
    M=1.0, kappa=1.0,
    c0=0.5
)

# 创建求解器
solver = CahnHilliardSolver(config)

# 初始化
solver.initialize_fields(seed=42)

# 运行模拟
result = solver.run(n_steps=5000)
```

### 电化学相场

```python
from phase_field.core.electrochemical import (
    ElectrochemicalPhaseField, ElectrochemicalConfig
)

config = ElectrochemicalConfig(
    nx=128, ny=128,
    temperature=298.15,
    E0=3.9,  # 开路电势
    applied_current=10.0  # A/m²
)

solver = ElectrochemicalPhaseField(config)
solver.initialize_fields()
result = solver.run()
```

### SEI生长模拟

```python
from phase_field.applications.sei_growth import SEIGrowthSimulator, SEIConfig

config = SEIConfig(
    nx=128, ny=64,
    component_names=['organic', 'Li2CO3', 'LiF'],
    sei_modulus=10.0,  # GPa
    include_mechanical_failure=True
)

simulator = SEIGrowthSimulator(config)
simulator.initialize_fields()
result = simulator.run()

# 获取SEI性质
sei_props = simulator.get_sei_properties()
print(f"SEI厚度: {sei_props['thickness']:.2f} nm")
```

### DFT耦合

```python
from phase_field.coupling.dft_coupling import DFTCoupling
from phase_field.workflow import PhaseFieldWorkflow

# 配置工作流
config = WorkflowConfig(
    dft_output_path="./vasp_results",
    run_dft=True,
    run_phase_field=True
)

# 运行工作流
workflow = PhaseFieldWorkflow(config)
results = workflow.run()
```

模块结构
--------

```
phase_field/
├── core/                    # 核心物理模型
│   ├── cahn_hilliard.py     # Cahn-Hilliard方程
│   ├── allen_cahn.py        # Allen-Cahn方程
│   ├── electrochemical.py   # 电化学相场
│   └── mechanochemistry.py  # 力学-化学耦合
├── solvers/                 # 数值求解器
│   ├── finite_difference.py # 有限差分
│   ├── finite_element.py    # 有限元
│   ├── gpu_solver.py        # GPU加速
│   ├── parallel_solver.py   # 并行求解
│   └── adaptive_mesh.py     # 自适应网格
├── coupling/                # 多尺度耦合
│   ├── dft_coupling.py      # DFT耦合
│   ├── md_coupling.py       # MD耦合
│   └── parameter_transfer.py # 参数传递
├── applications/            # 应用模块
│   ├── sei_growth.py        # SEI生长
│   ├── precipitation.py     # 沉淀相演化
│   ├── grain_boundary.py    # 晶界迁移
│   └── catalyst_reconstruction.py  # 催化剂重构
├── tests/                   # 测试套件
└── examples/                # 使用示例
```

性能优化
--------

### GPU加速

```python
from phase_field.solvers.gpu_solver import GPUSolver

gpu_solver = GPUSolver()

# 使用GPU加速FFT
laplacian = gpu_solver.laplacian_spectral(field, k_squared)

# 批量处理
results = gpu_solver.batch_process(fields, operation)
```

### 并行计算

```python
from phase_field.solvers.parallel_solver import ParallelSolver

parallel_solver = ParallelSolver(n_processes=4)
result = parallel_solver.run_domain_decomposition(solver, domain_size)
```

验证测试
--------

运行测试套件:

```bash
python -m phase_field.tests.test_models
```

或运行特定示例:

```bash
python phase_field/examples/example_sei.py
python phase_field/examples/example_precipitation.py
python phase_field/examples/example_workflow.py
```

参考文献
--------

1. Cahn, J.W. and Hilliard, J.E. (1958) Free energy of a nonuniform system.
2. Allen, S.M. and Cahn, J.W. (1979) A microscopic theory for antiphase boundary motion.
3. Guyre, D. et al. (2004) Phase field modeling of electrochemical systems.
4. Khachaturyan, A.G. (1983) Theory of structural transformations in solids.

作者
----
Phase Field Development Team
DFT-MD Coupling Project

许可证
------
MIT License
"""

__version__ = "1.0.0"
__author__ = "DFT-MD Coupling Team"

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}
