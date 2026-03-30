# 跨学科应用模块文档

## 概述

本模块将材料计算方法扩展到生物材料、地球科学和天体物理领域，实现了9个专业模块共计约3000行Python代码。

## 模块结构

```
dftlammps/
├── bio/                    # 生物材料
│   ├── protein_dft.py      # 蛋白质DFT计算 (600+行)
│   ├── dna_interaction.py  # DNA-材料相互作用 (600+行)
│   └── biomaterial_design.py  # 生物材料设计 (700+行)
├── geoscience/             # 地球科学
│   ├── mineral_physics.py  # 矿物物理 (600+行)
│   ├── geochemistry.py     # 地球化学模拟 (500+行)
│   └── mantle_convection.py # 地幔对流 (600+行)
└── astro/                  # 天体物理
    ├── interstellar_ice.py # 星际冰模拟 (500+行)
    ├── planetary_interior.py # 行星内部 (600+行)
    └── white_dwarf_crust.py # 白矮星壳层 (500+行)
```

## 应用案例

### 1. 药物-靶点相互作用

**应用场景**: 基于DFT计算的药物分子与蛋白质靶点结合能预测

```python
from dftlammps.bio.protein_dft import *

# 读取蛋白质结构
reader = ProteinStructureReader()
protein = reader.read_pdb("1hiv.pdb")

# 设置QM/MM计算
calculator = ProteinDFTCalculator(
    xc_functional=XCFunctionalBio.M06_2X,
    basis_set="6-31G(d,p)"
)

# 计算结合能
result = calculator.calculate_binding_energy(
    protein, ligand_coords, ligand_atoms, binding_site
)

print(f"结合能: {result.binding_energy * 627.509:.2f} kcal/mol")
```

**功能特性**:
- QM/MM混合计算方法
- 多种交换关联泛函支持
- 结合自由能计算
- BSSE校正
- 酶催化能垒分析

### 2. 地球深部物质

**应用场景**: 地幔矿物在极端温压条件下的物理性质计算

```python
from dftlammps.geoscience.mineral_physics import *

# 矿物数据库
db = MineralDatabase()
forsterite = db.get_mineral(MineralPhase.OLIVINE_ALPHA)

# 高温高压计算
calc = HighPressureCalculator(db)
velocities = calc.calculate_seismic_velocities(
    MineralPhase.OLIVINE_ALPHA, 
    pressure=20,  # GPa
    temperature=1500  # K
)

print(f"Vp = {velocities['Vp_km_s']:.2f} km/s")
```

**功能特性**:
- Birch-Murnaghan状态方程
- 地震波速计算
- 相变边界预测
- 弹性各向异性分析
- 地幔密度剖面

### 3. 中子星地壳

**应用场景**: 致密天体壳层的核物质和晶格物理

```python
from dftlammps.astro.white_dwarf_crust import *

# 中子星模型
ns = NeutronStarCrust(mass=1.4, radius=12)

# 构建壳层结构
crust = ns.build_crust_structure(n_layers=50)

# 物理参数
I_crust = ns.calculate_crustal_moment_of_inertia()
mu = ns.calculate_shear_modulus(rho=1e15, lattice_spacing=1e-14)

print(f"壳层转动惯量: {I_crust:.2e} kg·m²")
```

**功能特性**:
- 超密物质物态方程
- 库仑晶体性质
- 核燃烧过程模拟
- 壳层断裂应变估计
- 冷却演化轨迹

## API参考

### 生物材料模块

#### ProteinDFTCalculator
```python
class ProteinDFTCalculator:
    """蛋白质DFT计算器"""
    
    def __init__(self,
                 xc_functional: XCFunctionalBio,
                 basis_set: str,
                 solvation: SolvationModel,
                 qm_region: List[int])
    
    def calculate_binding_energy(self, ...) -> ProteinDFTResult
    def optimize_geometry(self, ...) -> Tuple[ProteinStructure, ProteinDFTResult]
    def calculate_spectroscopic_properties(self, ...) -> Dict
```

#### DNANanoparticleCalculator
```python
class DNANanoparticleCalculator:
    """DNA-纳米粒子相互作用计算器"""
    
    def calculate_adsorption_energy(self, ...) -> DNAAdsorptionResult
    def simulate_dynamics(self, ...) -> Dict
```

### 地球科学模块

#### MantleConvectionSolver
```python
class MantleConvectionSolver:
    """地幔对流求解器"""
    
    def __init__(self,
                 physical_params: PhysicalParameters,
                 rheology_params: RheologyParameters)
    
    def time_stepping(self, ...) -> List[ConvectionState]
    def calculate_nusselt_number(self, ...) -> float
```

#### PhaseTransitionCalculator
```python
class PhaseTransitionCalculator:
    """相变计算器"""
    
    def get_transition_pressure(self, ...) -> float
    def construct_phase_diagram(self, ...) -> Dict
```

### 天体物理模块

#### PlanetaryStructureModel
```python
class PlanetaryStructureModel:
    """行星结构模型"""
    
    def build_terrestrial_structure(self, ...) -> List[LayerProperties]
    def build_gas_giant_structure(self, ...) -> List[LayerProperties]
    def calculate_moment_of_inertia_factor(self) -> float
```

#### WhiteDwarfModel
```python
class WhiteDwarfModel:
    """白矮星模型"""
    
    def build_shell_structure(self, ...) -> List[ShellLayer]
    def calculate_crystallization_luminosity(self) -> float
```

## 运行示例

每个模块都包含独立的应用案例演示：

```bash
# 生物材料
python -m dftlammps.bio.protein_dft
python -m dftlammps.bio.dna_interaction
python -m dftlammps.bio.biomaterial_design

# 地球科学
python -m dftlammps.geoscience.mineral_physics
python -m dftlammps.geoscience.geochemistry
python -m dftlammps.geoscience.mantle_convection

# 天体物理
python -m dftlammps.astro.interstellar_ice
python -m dftlammps.astro.planetary_interior
python -m dftlammps.astro.white_dwarf_crust
```

## 技术特点

1. **统一的Python接口**: 所有模块遵循一致的API设计
2. **完整的类型注解**: 使用类型提示提高代码可维护性
3. **详细的文档字符串**: 每个类和函数都有完整文档
4. **丰富的应用案例**: 每个模块包含可运行的示例代码
5. **科学计算优化**: 基于NumPy的高效数值计算

## 引用

如果在研究中使用本模块，请引用：

```bibtex
@software{dftlammps_interdisciplinary,
  title = {DFT-LAMMPS Interdisciplinary Modules},
  year = {2025},
  note = {Biomaterials, Geoscience, and Astrophysics Applications}
}
```
