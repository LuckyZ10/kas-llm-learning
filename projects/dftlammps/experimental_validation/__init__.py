"""
DFT-LAMMPS 实验数据对接与验证模块

本模块提供计算结果与实验数据的对比验证功能，包括：
- 实验数据导入（CIF、POSCAR、JSON等格式）
- 数据库连接（Materials Project、AFLOW、COD等）
- 计算-实验对比分析
- 误差统计与不确定性量化
- 可视化报告生成
- 反馈优化循环

Phase 65: 实验数据对接与验证
"""

__version__ = "1.0.0"
__author__ = "DFT-LAMMPS Team"

# 数据格式
from .data_formats import (
    Lattice,
    AtomSite,
    CrystalStructure,
    ExperimentalProperty,
    ExperimentalDataset,
    DataFormat,
    CIFHandler,
    POSCARHandler,
    JSONHandler,
    read_structure,
    read_properties,
    write_structure,
    detect_format
)

# 数据源
from .data_sources import (
    DatabaseType,
    DatabaseConfig,
    SearchQuery,
    SearchResult,
    DatabaseConnector,
    MaterialsProjectConnector,
    AFLOWConnector,
    CODConnector,
    DatabaseManager,
    create_mp_connector,
    create_aflow_connector,
    create_cod_connector,
    create_default_manager
)

# 导入器
from .importers import (
    ImportConfig,
    ImportResult,
    DataImporter,
    FileImporter,
    CSVImporter,
    DatabaseImporter,
    BatchImporter,
    UnitConverter,
    PropertyNormalizer,
    import_from_file,
    import_from_database,
    batch_import,
    normalize_experimental_properties
)

# 对比分析
from .comparison import (
    ComparisonMetric,
    AgreementLevel,
    ComparisonResult,
    StatisticalAnalysis,
    ValidationReport,
    StructureComparator,
    PropertyComparator,
    UncertaintyQuantifier,
    BenchmarkSuite,
    compare_property,
    validate_properties,
    compare_structures
)

# 可视化
from .visualization import (
    PlotConfig,
    ValidationVisualizer,
    ReportGenerator,
    plot_parity,
    plot_validation_summary,
    generate_full_report
)

# 反馈优化
from .feedback import (
    OptimizationTarget,
    ParameterType,
    ParameterAdjustment,
    OptimizationRecommendation,
    FeedbackCycle,
    ErrorAnalyzer,
    ParameterOptimizer,
    DFTParameterOptimizer,
    MDParameterOptimizer,
    FeedbackLoop,
    AdaptiveLearningRate,
    create_feedback_loop,
    quick_optimize
)

__all__ = [
    # 数据格式
    'Lattice',
    'AtomSite',
    'CrystalStructure',
    'ExperimentalProperty',
    'ExperimentalDataset',
    'DataFormat',
    'CIFHandler',
    'POSCARHandler',
    'JSONHandler',
    'read_structure',
    'read_properties',
    'write_structure',
    'detect_format',
    
    # 数据源
    'DatabaseType',
    'DatabaseConfig',
    'SearchQuery',
    'SearchResult',
    'DatabaseConnector',
    'MaterialsProjectConnector',
    'AFLOWConnector',
    'CODConnector',
    'DatabaseManager',
    'create_mp_connector',
    'create_aflow_connector',
    'create_cod_connector',
    'create_default_manager',
    
    # 导入器
    'ImportConfig',
    'ImportResult',
    'DataImporter',
    'FileImporter',
    'CSVImporter',
    'DatabaseImporter',
    'BatchImporter',
    'UnitConverter',
    'PropertyNormalizer',
    'import_from_file',
    'import_from_database',
    'batch_import',
    'normalize_experimental_properties',
    
    # 对比分析
    'ComparisonMetric',
    'AgreementLevel',
    'ComparisonResult',
    'StatisticalAnalysis',
    'ValidationReport',
    'StructureComparator',
    'PropertyComparator',
    'UncertaintyQuantifier',
    'BenchmarkSuite',
    'compare_property',
    'validate_properties',
    'compare_structures',
    
    # 可视化
    'PlotConfig',
    'ValidationVisualizer',
    'ReportGenerator',
    'plot_parity',
    'plot_validation_summary',
    'generate_full_report',
    
    # 反馈优化
    'OptimizationTarget',
    'ParameterType',
    'ParameterAdjustment',
    'OptimizationRecommendation',
    'FeedbackCycle',
    'ErrorAnalyzer',
    'ParameterOptimizer',
    'DFTParameterOptimizer',
    'MDParameterOptimizer',
    'FeedbackLoop',
    'AdaptiveLearningRate',
    'create_feedback_loop',
    'quick_optimize'
]


def run_all_demos():
    """运行所有演示"""
    print("=" * 80)
    print("🚀 DFT-LAMMPS 实验数据验证模块 - 完整演示")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("📊 数据格式处理")
    print("=" * 80)
    from .data_formats import demo as data_demo
    data_demo()
    
    print("\n" + "=" * 80)
    print("🌐 数据源连接器")
    print("=" * 80)
    from .data_sources import demo as source_demo
    source_demo()
    
    print("\n" + "=" * 80)
    print("📥 数据导入器")
    print("=" * 80)
    from .importers import demo as import_demo
    import_demo()
    
    print("\n" + "=" * 80)
    print("🔬 对比分析")
    print("=" * 80)
    from .comparison import demo as compare_demo
    compare_demo()
    
    print("\n" + "=" * 80)
    print("📈 可视化")
    print("=" * 80)
    from .visualization import demo as viz_demo
    viz_demo()
    
    print("\n" + "=" * 80)
    print("🔄 反馈优化循环")
    print("=" * 80)
    from .feedback import demo as feedback_demo
    feedback_demo()
    
    print("\n" + "=" * 80)
    print("✅ 所有演示完成!")
    print("=" * 80)


def create_example_workflow():
    """创建示例工作流"""
    print("=" * 80)
    print("📝 示例工作流: 从实验数据导入到验证报告")
    print("=" * 80)
    
    # 步骤1: 导入实验数据
    print("\n步骤1: 导入实验数据")
    print("-" * 40)
    
    # 模拟从文件导入
    from .data_formats import CrystalStructure, Lattice, AtomSite, ExperimentalProperty
    
    structure = CrystalStructure(
        formula="SiO2",
        lattice=Lattice(4.9, 4.9, 5.4, 90, 90, 120),
        sites=[
            AtomSite('Si', 0.47, 0.0, 0.0),
            AtomSite('O', 0.41, 0.27, 0.12),
        ],
        space_group="P3221"
    )
    
    exp_dataset = ExperimentalDataset(
        structure=structure,
        properties=[
            ExperimentalProperty('band_gap', 8.9, 'eV', uncertainty=0.2, method='UV-Vis'),
            ExperimentalProperty('density', 2.65, 'g/cm^3', uncertainty=0.01, method='XRD')
        ],
        source_database='experiment',
        entry_id='SiO2_alpha'
    )
    
    print(f"✓ 导入实验数据: {exp_dataset}")
    
    # 步骤2: 模拟计算数据
    print("\n步骤2: 获取计算数据")
    print("-" * 40)
    
    comp_structure = CrystalStructure(
        formula="SiO2",
        lattice=Lattice(4.85, 4.85, 5.35, 90, 90, 120),
        sites=[
            AtomSite('Si', 0.469, 0.0, 0.0),
            AtomSite('O', 0.413, 0.268, 0.119),
        ],
        space_group="P3221"
    )
    
    comp_dataset = ExperimentalDataset(
        structure=comp_structure,
        properties=[
            ExperimentalProperty('band_gap', 8.5, 'eV', method='DFT-PBE'),
            ExperimentalProperty('density', 2.68, 'g/cm^3', method='DFT')
        ],
        source_database='DFT',
        entry_id='SiO2_calc'
    )
    
    print(f"✓ 计算数据: {comp_dataset}")
    
    # 步骤3: 对比分析
    print("\n步骤3: 计算-实验对比")
    print("-" * 40)
    
    from .comparison import PropertyComparator
    
    comparator = PropertyComparator()
    comparison_results = comparator.compare_datasets(comp_dataset, exp_dataset)
    
    print("对比结果:")
    for name, result in comparison_results.items():
        print(f"  • {result}")
    
    # 步骤4: 生成报告
    print("\n步骤4: 生成验证报告")
    print("-" * 40)
    
    from .comparison import validate_properties
    
    # 模拟更多数据点
    import numpy as np
    np.random.seed(42)
    
    exp_bg = [8.9, 3.4, 5.2, 6.1, 4.5]
    comp_bg = [8.5, 3.2, 5.0, 5.8, 4.3]
    uncertainties = [0.2, 0.1, 0.15, 0.12, 0.18]
    
    report = validate_properties(comp_bg, exp_bg, 'band_gap', uncertainties)
    
    print(f"✓ {report.get_summary()}")
    
    # 步骤5: 反馈优化建议
    print("\n步骤5: 生成优化建议")
    print("-" * 40)
    
    from .feedback import quick_optimize
    
    current_params = {'ENCUT': 400, 'KPOINTS': [4, 4, 4]}
    recommendations = quick_optimize(report, current_params, 'dft')
    
    if recommendations:
        print("优化建议:")
        for i, rec in enumerate(recommendations[:2], 1):
            print(f"\n  {i}. {rec.target.value} (优先级: {rec.priority})")
            for adj in rec.adjustments[:2]:
                print(f"     • {adj.parameter}: {adj.current_value} → {adj.suggested_value}")
    else:
        print("✓ 当前参数设置良好，无需优化")
    
    print("\n" + "=" * 80)
    print("✅ 示例工作流完成!")
    print("=" * 80)
    
    return {
        'experimental_data': exp_dataset,
        'computed_data': comp_dataset,
        'comparison': comparison_results,
        'report': report,
        'recommendations': recommendations
    }


if __name__ == '__main__':
    run_all_demos()
