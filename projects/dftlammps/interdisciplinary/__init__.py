"""
DFT-LAMMPS 跨学科应用模块
=========================

本模块扩展了DFT-LAMMPS平台到生物材料、地球科学和天体物理领域。

模块结构:
    bio/ - 生物材料应用
        - protein_dft.py: 蛋白质DFT计算
        - dna_interaction.py: DNA-材料相互作用
        - biomaterial_design.py: 生物材料设计
    
    geoscience/ - 地球科学应用
        - mineral_physics.py: 矿物物理
        - geochemistry.py: 地球化学模拟
        - mantle_convection.py: 地幔对流
    
    astro/ - 天体物理应用
        - interstellar_ice.py: 星际冰模拟
        - planetary_interior.py: 行星内部
        - white_dwarf_crust.py: 白矮星壳层

应用案例:
    - 药物-靶点相互作用
    - 地球深部物质
    - 中子星地壳
"""

__version__ = "1.0.0"
__author__ = "DFT-LAMMPS Team"

# Bio模块
from .bio.protein_dft import (
    ProteinDFTCalculator,
    ProteinStructure,
    ProteinStructureReader,
    EnzymeCatalysisAnalyzer,
    XCFunctionalBio,
    SolvationModel
)

from .bio.dna_interaction import (
    DNANanoparticleCalculator,
    DNAStructure,
    Nanoparticle,
    MaterialType,
    GeneDeliveryOptimizer,
    BiosensorDesigner
)

from .bio.biomaterial_design import (
    PolymerDesigner,
    BiomaterialDatabase,
    DegradationSimulator,
    CellMaterialInterface,
    ScaffoldDesigner,
    BiomaterialType,
    ApplicationTarget
)

# Geoscience模块
from .geoscience.mineral_physics import (
    HighPressureCalculator,
    PhaseTransitionCalculator,
    MantleComposition,
    MineralDatabase,
    MineralPhase,
    ElasticTensor
)

from .geoscience.geochemistry import (
    PartitionCoefficientDatabase,
    FractionalCrystallizationModel,
    FluidRockInteraction,
    IsotopeSystem,
    MagmaType
)

from .geoscience.mantle_convection import (
    MantleConvectionSolver,
    SubductionSimulator,
    MantlePlumeSimulator,
    PhysicalParameters,
    RheologyParameters,
    BoundaryCondition,
    RheologyLaw
)

# Astro模块
from .astro.interstellar_ice import (
    InterstellarIceSimulator,
    GrainSurface,
    IceLayer,
    IceComponent,
    IcePhase
)

from .astro.planetary_interior import (
    PlanetaryStructureModel,
    HighPressureEOS,
    CoreDynamics,
    MagneticDynamoModel,
    PlanetType,
    CoreComposition,
    MantleComposition
)

from .astro.white_dwarf_crust import (
    WhiteDwarfModel,
    NeutronStarCrust,
    NuclearBurningSimulator,
    CoolingModel,
    CompactObjectType,
    ShellComposition,
    CoulombCrystal
)

__all__ = [
    # Bio
    'ProteinDFTCalculator',
    'ProteinStructure',
    'DNANanoparticleCalculator',
    'DNAStructure',
    'GeneDeliveryOptimizer',
    'BiosensorDesigner',
    'PolymerDesigner',
    'ScaffoldDesigner',
    'BiomaterialType',
    'MaterialType',
    
    # Geoscience
    'HighPressureCalculator',
    'PhaseTransitionCalculator',
    'MantleComposition',
    'MineralDatabase',
    'MineralPhase',
    'PartitionCoefficientDatabase',
    'FractionalCrystallizationModel',
    'MantleConvectionSolver',
    'SubductionSimulator',
    'MantlePlumeSimulator',
    
    # Astro
    'InterstellarIceSimulator',
    'GrainSurface',
    'IceComponent',
    'PlanetaryStructureModel',
    'HighPressureEOS',
    'CoreDynamics',
    'MagneticDynamoModel',
    'WhiteDwarfModel',
    'NeutronStarCrust',
    'NuclearBurningSimulator',
    'CoolingModel',
    'PlanetType',
    'CompactObjectType',
]
