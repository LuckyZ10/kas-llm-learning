# 多尺度耦合模块技术文档

## 概述

本模块实现从原子尺度（DFT/MD）到介观尺度（相场）再到连续介质尺度的耦合模拟能力。

## 模块结构

```
dftlammps/multiscale/
├── phase_field.py          # 相场模拟
├── continuum.py            # 连续介质力学
├── parameter_passing.py    # 跨尺度参数传递
└── __init__.py            # 模块接口
```

## 1. 相场模拟 (phase_field.py)

### 1.1 核心功能

#### 枝晶生长模拟
基于Karma-Rappel相场模型，支持：
- 各向异性生长（四重、六重对称）
- 热扩散耦合
- 潜热释放
- 热噪声

```python
from dftlammps.multiscale import DendriteConfig, DendriteGrowthSolver

config = DendriteConfig(
    dimensions=2,
    nx=256, ny=256,
    dx=2.0,  # nm
    interface_width=5.0,
    anisotropy_strength=0.05,
    undercooling=0.2
)

solver = DendriteGrowthSolver(config)
solver.initialize_phi(pattern="nucleus", radius=10.0)
solver.run(n_steps=50000)
```

#### 相分离动力学 (Cahn-Hilliard)
```python
from dftlammps.multiscale import SpinodalConfig, SpinodalDecompositionSolver

config = SpinodalConfig(
    initial_composition=0.5,
    chi_parameter=3.0,
    mobility_c=1.0
)

solver = SpinodalDecompositionSolver(config)
solver.run()
domain_size = solver.get_domain_size()
```

### 1.2 与MD的耦合

从MD提取相场参数：

```python
from dftlammps.multiscale import MDtoPhaseFieldExtractor

extractor = MDtoPhaseFieldExtractor()
extractor.load_md_trajectory("md_trajectory.xyz")

# 提取扩散系数
D = extractor.extract_diffusion_coefficient(temperature=300.0)

# 提取界面能
gamma = extractor.extract_interface_energy(
    solid_phase_indices=range(0, 500),
    liquid_phase_indices=range(500, 1000),
    interface_area=100.0  # Å²
)

# 获取参数对象
params = extractor.get_params()
params.save_params("pf_params.json")
```

### 1.3 外部框架接口

#### PRISMS-PF接口
```python
from dftlammps.multiscale import PRISMSPFInterface

prisms = PRISMSPFInterface(prisms_pf_path="/path/to/prismspf")
prisms.generate_input_file(config, model_name="allen_cahn")
prisms.run_simulation(input_file="parameters.prm", n_procs=4)
```

#### MOOSE接口
```python
from dftlammps.multiscale import MOOSEInterface

moose = MOOSEInterface(moose_path="/path/to/moose")
moose.generate_input_file(config, problem_type="dendrite")
moose.run_simulation(input_file="moose_input.i", n_procs=8)
```

## 2. 连续介质力学 (continuum.py)

### 2.1 有限元求解器

#### 热传导分析
```python
from dftlammps.multiscale import (
    FEMConfig, ThermalConfig, 
    FEMMesh, ThermalSolver, MaterialModel
)

# 创建网格
fem_config = FEMConfig(
    dimensions=2,
    lx=100, ly=100,  # nm
    nx=50, ny=50,
    element_type="quad"
)
mesh = FEMMesh(fem_config)

# 材料属性
material = MaterialModel()
material.thermal.thermal_conductivity = 100  # W/(m·K)

# 求解
thermal_config = ThermalConfig(
    bc_temperature=[
        {'boundary': 'left', 'value': 400},
        {'boundary': 'right', 'value': 300}
    ]
)

solver = ThermalSolver(mesh, material, thermal_config)
T = solver.solve_steady()

# 计算热流
heat_flux = solver.compute_heat_flux()
```

#### 力学分析
```python
from dftlammps.multiscale import MechanicsConfig, MechanicsSolver

mechanics_config = MechanicsConfig(
    bc_displacement=[
        {'boundary': 'left', 'value': 0.0, 'direction': 0},
        {'boundary': 'bottom', 'value': 0.0, 'direction': 1}
    ],
    bc_traction=[
        {'boundary': 'right', 'traction': [100.0, 0.0]}
    ]
)

solver = MechanicsSolver(mesh, material, mechanics_config)
displacement = solver.solve_static()
von_mises = solver.get_von_mises_stress()
```

### 2.2 耦合热-力分析
```python
from dftlammps.multiscale import CoupledConfig, CoupledThermalMechanicsSolver

coupled_config = CoupledConfig(
    mechanics=mechanics_config,
    thermal=thermal_config,
    coupling_scheme="staggered",
    dt=0.001,
    n_steps=1000
)

solver = CoupledThermalMechanicsSolver(mesh, material, coupled_config)
results = solver.solve()
```

### 2.3 与DFT的耦合

从DFT提取弹性常数：

```python
from dftlammps.multiscale.parameter_passing import DFTParameterExtractor

extractor = DFTParameterExtractor()
elastic = extractor.extract_elastic_constants("POSCAR")

# 使用在连续介质模型中
material.elastic.C11 = elastic.C11
material.elastic.C12 = elastic.C12
material.elastic.C44 = elastic.C44
```

## 3. 跨尺度参数传递 (parameter_passing.py)

### 3.1 完整工作流

```python
from dftlammps.multiscale import ParameterPassingWorkflow

workflow = ParameterPassingWorkflow()

# 从DFT提取
workflow.extract_from_dft(
    structure_file="POSCAR",
    calculation_type="elastic"
)

# 从MD提取
workflow.extract_from_md(
    trajectory_file="md.xyz",
    extraction_type="diffusion"
)

# 参数转换
params = workflow.convert_parameters()

# 导出
workflow.export_parameters("multiscale_params.json")
```

### 3.2 参数转换

#### 弹性常数 → 连续介质
```python
from dftlammps.multiscale import ElasticConstants, ParameterConverter

elastic = ElasticConstants(C11=100, C12=50, C44=50)
converter = ParameterConverter()
continuum_params = converter.elastic_to_continuum(elastic)
# 结果: {'E': ..., 'nu': ..., 'G': ...}
```

#### 扩散系数 → 相场
```python
transport = TransportProperties(D=1e-9, activation_energy=0.5)
pf_params = converter.diffusion_to_phase_field(transport)
```

#### 界面性质 → 相场
```python
interface = InterfaceProperties(
    gamma=0.3,  # J/m²
    interface_width=2.0,  # nm
    mobility=1.5e-10
)
pf_params = converter.interface_to_phase_field(interface)
```

### 3.3 参数验证

```python
from dftlammps.multiscale import ParameterValidator

validator = ParameterValidator()

# 验证弹性稳定性
validation = validator.validate_elastic_constants(elastic)
# 检查Born稳定性条件

# 验证扩散系数范围
validation = validator.validate_diffusion_coefficient(
    transport,
    expected_range=(1e-20, 1e-8)
)

# 跨尺度一致性检查
consistency = validator.cross_scale_consistency_check(multiscale_params)
```

## 4. 应用案例

### 4.1 枝晶生长多尺度模拟

详见: `dftlammps/applications/dendrite_growth/case_dendrite_multiscale.py`

```bash
python -m dftlammps.applications.dendrite_growth.case_dendrite_multiscale \
    --working-dir ./dendrite_simulation
```

### 4.2 SEI界面演化

详见: `dftlammps/applications/solid_electrolyte_interface/case_sei_interface.py`

```bash
python -m dftlammps.applications.solid_electrolyte_interface.case_sei_interface \
    --working-dir ./sei_evolution
```

## 5. 物理模型

### 5.1 相场方程

#### Karma-Rappel模型（枝晶生长）

$$\tau(\theta)\frac{\partial\phi}{\partial t} = -\frac{\delta F}{\delta\phi}$$

其中自由能泛函：

$$F = \int d\mathbf{r} \left[ \frac{W^2(\theta)}{2}|\nabla\phi|^2 + f(\phi, T) \right]$$

#### Cahn-Hilliard方程（相分离）

$$\frac{\partial c}{\partial t} = \nabla \cdot \left(M \nabla \frac{\delta F}{\delta c}\right)$$

化学势：

$$\mu = \frac{\partial f}{\partial c} - \kappa \nabla^2 c$$

### 5.2 连续介质方程

#### 热传导

$$\rho C_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q$$

#### 线弹性力学

$$\nabla \cdot \boldsymbol{\sigma} + \mathbf{f} = 0$$

$$\boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}$$

## 6. 性能优化

### 6.1 并行计算

```python
# PRISMS-PF并行
prisms.run_simulation(input_file="params.prm", n_procs=16)

# MOOSE并行
moose.run_simulation(input_file="input.i", n_procs=16)
```

### 6.2 网格自适应

```python
fem_config = FEMConfig(
    mesh_type="adaptive",
    refinement_strategy="gradient_based"
)
```

## 7. 输出和可视化

### 7.1 VTK导出
```python
mesh.export_to_vtk(
    "result.vtu",
    temperature=T,
    displacement=displacement[:, 0],
    von_mises=von_mises
)
```

### 7.2 NumPy数组
```python
# 相场数据
np.save("phi.npy", solver.phi)
np.save("temperature.npy", solver.T)

# 连续介质数据
np.save("displacement.npy", solver.displacement)
np.save("stress.npy", solver.stress)
```

## 8. 参考文献

1. Karma, A., & Rappel, W. J. (1998). Quantitative phase-field modeling of dendritic growth. Physical Review E, 57(4), 4323.

2. Cahn, J. W., & Hilliard, J. E. (1958). Free energy of a nonuniform system. I. Interfacial free energy. The Journal of Chemical Physics, 28(2), 258-267.

3. Steinbach, I. (2009). Phase-field models in materials science. Modelling and Simulation in Materials Science and Engineering, 17(7), 073001.

4. Provatas, N., & Elder, K. (2010). Phase-Field Methods in Materials Science and Engineering. Wiley-VCH.

5. Hughes, T. J. (2012). The Finite Element Method: Linear Static and Dynamic Finite Element Analysis. Courier Corporation.
