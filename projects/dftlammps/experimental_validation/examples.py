#!/usr/bin/env python
"""
实验数据验证模块使用示例
展示如何导入实验数据、进行对比分析和生成报告
"""
import numpy as np
from pathlib import Path

from dftlammps.experimental_validation import (
    # 数据格式
    Lattice, AtomSite, CrystalStructure, ExperimentalProperty, ExperimentalDataset,
    
    # 导入
    ImportConfig, FileImporter, UnitConverter, PropertyNormalizer,
    
    # 对比
    ComparisonResult, ValidationReport, PropertyComparator, validate_properties,
    compare_property, StructureComparator, BenchmarkSuite,
    
    # 可视化
    PlotConfig, ReportGenerator, generate_full_report,
    
    # 反馈
    FeedbackLoop, quick_optimize, create_feedback_loop
)


def example_1_basic_data_handling():
    """示例1: 基础数据处理"""
    print("=" * 80)
    print("示例1: 基础数据处理")
    print("=" * 80)
    
    # 创建晶格
    lattice = Lattice(a=5.43, b=5.43, c=5.43, alpha=90, beta=90, gamma=90)
    print(f"晶格体积: {lattice.volume:.4f} Å³")
    
    # 创建晶体结构
    sites = [
        AtomSite('Si', 0.0, 0.0, 0.0),
        AtomSite('Si', 0.25, 0.25, 0.25),
        AtomSite('Si', 0.5, 0.0, 0.5),
        AtomSite('Si', 0.75, 0.25, 0.75),
        AtomSite('Si', 0.0, 0.5, 0.5),
        AtomSite('Si', 0.25, 0.75, 0.75),
        AtomSite('Si', 0.5, 0.5, 0.0),
        AtomSite('Si', 0.75, 0.75, 0.25),
    ]
    
    structure = CrystalStructure(
        formula='Si',
        lattice=lattice,
        sites=sites,
        space_group='Fd-3m',
        space_group_number=227
    )
    
    print(f"结构: {structure}")
    print(f"原子数: {structure.num_atoms}")
    print(f"元素: {structure.elements}")
    print(f"组成: {structure.composition}")
    
    # 创建实验属性
    properties = [
        ExperimentalProperty(
            name='band_gap',
            value=1.12,
            unit='eV',
            uncertainty=0.01,
            temperature=300,
            method='Optical absorption',
            reference='Green 1990'
        ),
        ExperimentalProperty(
            name='lattice_constant',
            value=5.4307,
            unit='Å',
            uncertainty=0.0001,
            temperature=298,
            method='XRD',
            reference='COD 9008566'
        ),
        ExperimentalProperty(
            name='bulk_modulus',
            value=97.8,
            unit='GPa',
            uncertainty=2.0,
            temperature=300,
            method='Ultrasonic'
        )
    ]
    
    print("\n实验属性:")
    for prop in properties:
        print(f"  • {prop}")
    
    # 创建数据集
    dataset = ExperimentalDataset(
        structure=structure,
        properties=properties,
        source_database='COD',
        entry_id='9008566',
        metadata={'quality': 'high', 'peer_reviewed': True}
    )
    
    print(f"\n数据集: {dataset}")
    
    return dataset


def example_2_comparison_analysis():
    """示例2: 对比分析"""
    print("\n" + "=" * 80)
    print("示例2: 计算-实验对比分析")
    print("=" * 80)
    
    np.random.seed(42)
    
    # 模拟计算值和实验值
    n_samples = 30
    true_values = np.random.uniform(1, 10, n_samples)
    experimental = true_values + np.random.normal(0, 0.1, n_samples)
    uncertainties = np.random.uniform(0.05, 0.2, n_samples)
    
    # 模拟不同质量的计算
    computed_good = true_values + np.random.normal(0, 0.2, n_samples)
    computed_poor = true_values + np.random.normal(0.5, 0.4, n_samples)
    
    # 创建对比结果
    print("\n高质量计算 vs 实验:")
    results_good = [
        compare_property(c, e, u, 'band_gap')
        for c, e, u in zip(computed_good, experimental, uncertainties)
    ]
    
    report_good = validate_properties(
        computed_good.tolist(),
        experimental.tolist(),
        'band_gap',
        uncertainties.tolist()
    )
    
    if report_good.statistics:
        print(f"  MAE: {report_good.statistics.mae:.4f}")
        print(f"  MAPE: {report_good.statistics.mape:.2f}%")
        print(f"  R²: {report_good.statistics.r2:.4f}")
        print(f"  验证得分: {report_good.validation_score:.1f}/100")
        print(f"  通过验证: {'是' if report_good.is_validated else '否'}")
    
    print("\n低质量计算 vs 实验:")
    results_poor = [
        compare_property(c, e, u, 'band_gap')
        for c, e, u in zip(computed_poor, experimental, uncertainties)
    ]
    
    report_poor = validate_properties(
        computed_poor.tolist(),
        experimental.tolist(),
        'band_gap',
        uncertainties.tolist()
    )
    
    if report_poor.statistics:
        print(f"  MAE: {report_poor.statistics.mae:.4f}")
        print(f"  MAPE: {report_poor.statistics.mape:.2f}%")
        print(f"  R²: {report_poor.statistics.r2:.4f}")
        print(f"  验证得分: {report_poor.validation_score:.1f}/100")
        print(f"  通过验证: {'是' if report_poor.is_validated else '否'}")
    
    return report_good, report_poor


def example_3_benchmark():
    """示例3: 基准测试"""
    print("\n" + "=" * 80)
    print("示例3: 多属性基准测试")
    print("=" * 80)
    
    suite = BenchmarkSuite()
    
    np.random.seed(42)
    
    properties = ['band_gap', 'lattice_constant', 'bulk_modulus', 'shear_modulus']
    
    for prop in properties:
        n = np.random.randint(15, 40)
        
        # 生成模拟数据
        if 'lattice' in prop:
            true = np.random.uniform(3, 8, n)
        elif 'modulus' in prop:
            true = np.random.uniform(10, 200, n)
        else:
            true = np.random.uniform(0.5, 10, n)
        
        exp = true + np.random.normal(0, 0.03 * np.std(true), n)
        comp = true + np.random.normal(0, 0.08 * np.std(true), n)
        
        report = validate_properties(comp.tolist(), exp.tolist(), prop)
        suite.add_benchmark(prop, report)
    
    print("\n基准测试结果:")
    print(suite.get_summary_table())
    print(f"\n总体得分: {suite.overall_score():.1f}/100")
    
    # 最佳表现
    best = suite.get_best_performing('mape')
    print(f"\n最佳表现 (MAPE):")
    for name, mape in best[:3]:
        print(f"  {name}: {mape:.2f}%")
    
    return suite


def example_4_unit_conversion():
    """示例4: 单位转换"""
    print("\n" + "=" * 80)
    print("示例4: 单位转换")
    print("=" * 80)
    
    conversions = [
        (1, 'eV', 'J'),
        (5, 'GPa', 'Pa'),
        (300, 'K', 'C'),
        (4.2, 'Å', 'nm'),
        (100, 'MPa', 'GPa'),
    ]
    
    print("\n单位转换示例:")
    for value, from_u, to_u in conversions:
        result = UnitConverter.convert(value, from_u, to_u)
        print(f"  {value} {from_u} = {result:.6e} {to_u}")
    
    # 属性标准化
    print("\n属性名称标准化:")
    names = ['bandgap', 'Band Gap', 'E_g', 'formation_energy', 'E_form']
    for name in names:
        normalized = PropertyNormalizer.normalize_name(name)
        unit = PropertyNormalizer.get_standard_unit(normalized)
        print(f"  {name:20s} → {normalized:20s} (标准单位: {unit or 'N/A'})")


def example_5_feedback_optimization():
    """示例5: 反馈优化"""
    print("\n" + "=" * 80)
    print("示例5: 反馈优化循环")
    print("=" * 80)
    
    # 创建反馈循环
    loop = create_feedback_loop()
    
    # 模拟验证报告
    np.random.seed(42)
    results = []
    for i in range(20):
        comp = 5.0 + np.random.normal(0.3, 0.4)
        exp = 5.0 + np.random.normal(0, 0.15)
        results.append(compare_property(comp, exp, 0.1, 'band_gap'))
    
    report = validate_properties(
        [r.computed_value for r in results],
        [r.experimental_value for r in results],
        'band_gap',
        [0.1] * len(results)
    )
    
    print(f"\n初始验证状态:")
    if report.statistics:
        print(f"  MAPE: {report.statistics.mape:.2f}%")
        print(f"  系统偏差: {report.statistics.systematic_error:.2f}%")
    
    # 当前参数
    current_params = {
        'ENCUT': 400,
        'KPOINTS': [4, 4, 4],
        'SIGMA': 0.2,
        'ISMEAR': 0
    }
    
    print(f"\n当前DFT参数:")
    for param, value in current_params.items():
        print(f"  {param}: {value}")
    
    # 运行反馈循环
    cycle = loop.run_cycle(report, current_params, 'dft')
    
    print(f"\n优化建议:")
    for i, rec in enumerate(cycle.recommendations[:3], 1):
        print(f"\n  {i}. 目标: {rec.target.value}, 优先级: {rec.priority}")
        print(f"     预期效果: {rec.expected_outcome}")
        print(f"     参数调整:")
        for adj in rec.adjustments[:2]:
            print(f"       • {adj.parameter}: {adj.current_value} → {adj.suggested_value}")
    
    # 模拟优化后的结果
    print("\n模拟优化后的验证:")
    improved_results = []
    for i in range(20):
        comp = 5.0 + np.random.normal(0.1, 0.25)  # 误差减小
        exp = 5.0 + np.random.normal(0, 0.15)
        improved_results.append(compare_property(comp, exp, 0.1, 'band_gap'))
    
    new_report = validate_properties(
        [r.computed_value for r in improved_results],
        [r.experimental_value for r in improved_results],
        'band_gap',
        [0.1] * len(improved_results)
    )
    
    if new_report.statistics:
        print(f"  新MAPE: {new_report.statistics.mape:.2f}%")
    
    improvement = loop.verify_improvement(cycle, new_report)
    print(f"  改善程度: {improvement:.1f}%")
    print(f"  状态: {cycle.status}")


def example_6_report_generation():
    """示例6: 报告生成"""
    print("\n" + "=" * 80)
    print("示例6: 报告生成")
    print("=" * 80)
    
    # 创建多个验证报告
    reports = []
    np.random.seed(42)
    
    for prop in ['band_gap', 'lattice_constant', 'bulk_modulus']:
        n = np.random.randint(20, 35)
        true = np.random.uniform(1, 100, n)
        exp = true + np.random.normal(0, 0.05 * np.std(true), n)
        comp = true + np.random.normal(0, 0.08 * np.std(true), n)
        
        report = validate_properties(comp.tolist(), exp.tolist(), prop)
        reports.append(report)
    
    # 生成报告
    generator = ReportGenerator()
    for report in reports:
        generator.add_report(report)
    
    # HTML报告
    html_path = '/tmp/example_validation_report.html'
    try:
        generator.generate_html_report(html_path)
        print(f"✓ HTML报告已生成: {html_path}")
    except Exception as e:
        print(f"✗ HTML报告生成失败: {e}")
    
    # Markdown报告
    md_path = '/tmp/example_validation_report.md'
    try:
        generator.generate_markdown_report(md_path)
        print(f"✓ Markdown报告已生成: {md_path}")
    except Exception as e:
        print(f"✗ Markdown报告生成失败: {e}")


def main():
    """运行所有示例"""
    print("=" * 80)
    print("🔬 DFT-LAMMPS 实验数据验证模块 - 使用示例")
    print("=" * 80)
    
    # 运行示例
    example_1_basic_data_handling()
    example_2_comparison_analysis()
    example_3_benchmark()
    example_4_unit_conversion()
    example_5_feedback_optimization()
    example_6_report_generation()
    
    print("\n" + "=" * 80)
    print("✅ 所有示例完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
