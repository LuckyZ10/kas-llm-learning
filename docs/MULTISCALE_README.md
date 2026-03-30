# DFT+LAMMPS 多尺度耦合模块

## 模块概览

本扩展为DFT+LAMMPS平台添加了多尺度耦合能力，实现了从原子尺度到连续介质尺度的无缝集成。

## 新增模块

### 1. dftlammps/multiscale/ - 多尺度耦合模块

#### phase_field.py - 相场模拟
- **枝晶生长求解器**: Karma-Rappel相场模型，支持各向异性生长
- **相分离求解器**: Cahn-Hilliard方程，模拟旋节分解
- **MD耦合**: 从分子动力学提取界面能、扩散系数
- **外部接口**: PRISMS-PF和MOOSE框架支持

#### continuum.py - 连续介质力学
- **有限元网格**: 2D/3D结构化网格生成
- **热传导求解器**: 稳态/瞬态热分析
- **力学求解器**: 线弹性有限元分析
- **耦合求解器**: 热-力耦合问题
- **FEniCS接口**: 高级有限元功能

#### parameter_passing.py - 跨尺度参数传递
- **DFT参数提取**: 弹性常数、界面能、表面能
- **MD参数提取**: 扩散系数、界面迁移率
- **自动转换**: 原子尺度 → 介观/连续尺度
- **参数验证**: 稳定性条件、一致性检查

### 2. dftlammps/advanced_potentials/ - 先进ML势接口

#### mace_interface.py - MACE高阶等变势
- 严格的旋转等变性
- 能量、力、应力预测
- 主动学习工作流
- LAMMPS集成

#### chgnet_interface.py - CHGNet图神经网络势
- 磁矩预测
- 电荷密度信息
- 高通量磁性材料筛选
- DFT预筛选

#### orb_interface.py - Orb超快推理势
- 每秒数百万原子处理能力
- 大规模MD模拟
- 高通量筛选优化
- 性能基准测试

#### __init__.py - 统一接口
- 势选择器（自动推荐最佳模型）
- 统一API（跨模型一致接口）
- 能力查询（检查支持的特性）

### 3. 应用案例

#### dendrite_growth/case_dendrite_multiscale.py
金属锂枝晶生长的完整多尺度模拟：
1. DFT计算界面能和弹性常数
2. MD模拟扩散系数和界面迁移率
3. 参数传递到相场模型
4. 枝晶生长动力学模拟
5. 连续介质热应力分析

#### solid_electrolyte_interface/case_sei_interface.py
SEI界面演化分析：
1. DFT分析界面电子结构和反应能
2. MD模拟离子传输
3. 相场模拟SEI层生长
4. 力学稳定性评估
5. 电池性能预测

## 快速入门

### 相场模拟

```python
from dftlammps.multiscale import PhaseFieldWorkflow, DendriteConfig

# 创建工作流
workflow = PhaseFieldWorkflow()

# 从MD提取参数
workflow.extract_parameters_from_md("md_trajectory.xyz")

# 设置并运行模拟
workflow.setup_dendrite_simulation()
workflow.run_simulation()

# 分析结果
results = workflow.analyze_results()
```

### 连续介质分析

```python
from dftlammps.multiscale import ContinuumWorkflow, FEMConfig

workflow = ContinuumWorkflow()

# 设置材料（从DFT参数）
workflow.setup_material(
    elastic_props={'C11': 100, 'C12': 50, 'C44': 50}
)

# 生成网格
workflow.generate_mesh(FEMConfig(nx=50, ny=50))

# 运行分析
results = workflow.run_mechanics_analysis(mechanics_config)
```

### 先进ML势

```python
from dftlammps.advanced_potentials import load_ml_potential

# 加载MACE
mace = load_ml_potential("mace", model_name="medium")

# 加载CHGNet（磁性材料）
chgnet = load_ml_potential("chgnet")

# 加载Orb（高通量筛选）
orb = load_ml_potential("orb", device="cuda")

# 统一使用
atoms.calc = mace
energy = atoms.get_potential_energy()
```

## 技术文档

详细文档位于 `docs/` 目录：

- `multiscale_technical_documentation.md` - 多尺度模块完整文档
- `advanced_potentials_technical_documentation.md` - ML势接口文档

## 依赖要求

### 基础依赖
- numpy >= 1.20
- scipy >= 1.7
- ase >= 3.22
- pymatgen >= 2022

### 可选依赖
- **相场外部框架**: PRISMS-PF, MOOSE
- **有限元**: FEniCS/dolfinx, scikit-fem
- **MACE**: mace-torch
- **CHGNet**: chgnet
- **Orb**: orb-models

### 安装可选依赖

```bash
# MACE
pip install mace-torch

# CHGNet
pip install chgnet

# Orb
pip install orb-models

# FEniCS
pip install fenics-dolfinx
```

## 文件清单

```
dftlammps/
├── multiscale/
│   ├── __init__.py
│   ├── phase_field.py          # 相场模拟 (45KB)
│   ├── continuum.py            # 连续介质力学 (50KB)
│   └── parameter_passing.py    # 参数传递 (37KB)
│
├── advanced_potentials/
│   ├── __init__.py             # 统一接口 (19KB)
│   ├── mace_interface.py       # MACE接口 (29KB)
│   ├── chgnet_interface.py     # CHGNet接口 (26KB)
│   └── orb_interface.py        # Orb接口 (30KB)
│
applications/
├── dendrite_growth/
│   └── case_dendrite_multiscale.py    # 枝晶生长案例 (21KB)
│
└── solid_electrolyte_interface/
    └── case_sei_interface.py          # SEI界面案例 (25KB)

docs/
├── multiscale_technical_documentation.md      # 多尺度文档 (8KB)
└── advanced_potentials_technical_documentation.md  # ML势文档 (10KB)
```

## 总计代码量

- **多尺度模块**: ~132KB Python代码
- **ML势接口**: ~105KB Python代码
- **应用案例**: ~46KB Python代码
- **技术文档**: ~18KB Markdown

**总计**: ~300KB 新增代码和文档

## 关键特性

### 1. 无缝尺度桥接
- 自动参数提取和转换
- 单位系统自动处理
- 不确定性量化

### 2. 多求解器支持
- 内置Python求解器（快速原型）
- PRISMS-PF/MOOSE（生产级）
- FEniCS（高级有限元）

### 3. 先进的ML势
- 统一API访问多个模型
- 自动势选择
- 性能优化（GPU/批处理）

### 4. 生产就绪
- 完整的错误处理
- 详细日志记录
- 命令行工具

## 应用示例

### 锂电池研究
```bash
# 枝晶生长多尺度模拟
python -m dftlammps.applications.dendrite_growth.case_dendrite_multiscale \
    --working-dir ./dendrite_sim

# SEI界面演化分析
python -m dftlammps.applications.solid_electrolyte_interface.case_sei_interface \
    --working-dir ./sei_analysis
```

### 高通量筛选
```python
from dftlammps.advanced_potentials import OrbWorkflow, OrbConfig

# 使用Orb进行超快筛选
orb = OrbWorkflow(OrbConfig(batch_size=128))
selected, results = orb.batch_screening(candidates)
```

### 磁性材料发现
```python
from dftlammps.advanced_potentials import CHGNetWorkflow

# 筛选磁性材料
chgnet = CHGNetWorkflow(CHGNetConfig(predict_magmom=True))
magnetic_materials = chgnet.batch_predict(structures)
```

## 未来扩展

计划添加的功能：
- 流体动力学耦合（两相流）
- 电磁场模拟
- 更多ML势支持（Equiformer, TorchMD等）
- 自动工作流编排

## 参考文献

1. Karma & Rappel, Phys. Rev. E (1998) - 枝晶生长相场模型
2. Cahn & Hilliard, J. Chem. Phys. (1958) - 相分离理论
3. Batatia et al., arXiv:2206.07697 - MACE
4. Deng et al., Nature Mach. Intell. (2023) - CHGNet
5. Stärk et al., arXiv:2410.22581 - Orb
