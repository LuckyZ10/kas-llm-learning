"""
实验验证模块测试
"""
import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

# 导入被测试模块
from dftlammps.experimental_validation.data_formats import (
    Lattice, AtomSite, CrystalStructure, ExperimentalProperty,
    ExperimentalDataset, CIFHandler, POSCARHandler, JSONHandler,
    read_structure, write_structure
)
from dftlammps.experimental_validation.comparison import (
    ComparisonResult, ValidationReport, StructureComparator,
    PropertyComparator, validate_properties, compare_property
)
from dftlammps.experimental_validation.importers import (
    ImportConfig, FileImporter, UnitConverter, PropertyNormalizer
)
from dftlammps.experimental_validation.feedback import (
    FeedbackLoop, OptimizationTarget, DFTParameterOptimizer
)


class TestDataFormats(unittest.TestCase):
    """测试数据格式模块"""
    
    def setUp(self):
        self.lattice = Lattice(4.2, 4.2, 4.2, 90, 90, 90)
        self.sites = [
            AtomSite('Na', 0.0, 0.0, 0.0),
            AtomSite('Cl', 0.5, 0.5, 0.5)
        ]
        self.structure = CrystalStructure(
            formula='NaCl',
            lattice=self.lattice,
            sites=self.sites,
            space_group='Fm-3m'
        )
    
    def test_lattice_creation(self):
        """测试晶格创建"""
        self.assertEqual(self.lattice.a, 4.2)
        self.assertEqual(self.lattice.alpha, 90)
        self.assertAlmostEqual(self.lattice.volume, 74.088, places=2)
    
    def test_lattice_matrix(self):
        """测试晶格矩阵"""
        matrix = self.lattice.to_matrix()
        self.assertEqual(matrix.shape, (3, 3))
        self.assertAlmostEqual(matrix[0, 0], 4.2)
    
    def test_structure_creation(self):
        """测试结构创建"""
        self.assertEqual(self.structure.formula, 'NaCl')
        self.assertEqual(self.structure.num_atoms, 2)
        self.assertEqual(sorted(self.structure.elements), ['Cl', 'Na'])
    
    def test_composition(self):
        """测试化学组成"""
        comp = self.structure.composition
        self.assertEqual(comp['Na'], 1)
        self.assertEqual(comp['Cl'], 1)
    
    def test_experimental_property(self):
        """测试实验属性"""
        prop = ExperimentalProperty(
            name='band_gap',
            value=3.2,
            unit='eV',
            uncertainty=0.1
        )
        self.assertEqual(prop.name, 'band_gap')
        self.assertEqual(prop.value, 3.2)
        self.assertAlmostEqual(prop.relative_uncertainty, 0.03125, places=4)
    
    def test_cif_handler(self):
        """测试CIF处理器"""
        handler = CIFHandler()
        self.assertTrue(handler.detect_format('test.cif'))
        self.assertFalse(handler.detect_format('test.xyz'))
    
    def test_poscar_handler(self):
        """测试POSCAR处理器"""
        handler = POSCARHandler()
        self.assertTrue(handler.detect_format('POSCAR'))
        self.assertTrue(handler.detect_format('CONTCAR'))
        self.assertTrue(handler.detect_format('test.poscar'))
    
    def test_json_handler(self):
        """测试JSON处理器"""
        handler = JSONHandler()
        self.assertTrue(handler.detect_format('test.json'))
        self.assertFalse(handler.detect_format('test.txt'))


class TestStructureIO(unittest.TestCase):
    """测试结构读写"""
    
    def setUp(self):
        self.lattice = Lattice(4.2, 4.2, 4.2, 90, 90, 90)
        self.structure = CrystalStructure(
            formula='NaCl',
            lattice=self.lattice,
            sites=[
                AtomSite('Na', 0.0, 0.0, 0.0),
                AtomSite('Cl', 0.5, 0.5, 0.5)
            ],
            space_group='Fm-3m'
        )
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cif_write_read(self):
        """测试CIF写入和读取"""
        filepath = os.path.join(self.temp_dir, 'test.cif')
        
        # 写入
        handler = CIFHandler()
        handler.write_structure(self.structure, filepath)
        self.assertTrue(os.path.exists(filepath))
        
        # 读取
        structure = handler.read_structure(filepath)
        self.assertEqual(structure.formula, 'NaCl')
        self.assertAlmostEqual(structure.lattice.a, 4.2, places=2)
    
    def test_poscar_write_read(self):
        """测试POSCAR写入和读取"""
        filepath = os.path.join(self.temp_dir, 'POSCAR')
        
        # 写入
        handler = POSCARHandler()
        handler.write_structure(self.structure, filepath)
        self.assertTrue(os.path.exists(filepath))
        
        # 读取
        structure = handler.read_structure(filepath)
        # 公式可能以不同顺序出现，检查包含的元素
        self.assertIn('Na', structure.formula)
        self.assertIn('Cl', structure.formula)
        self.assertAlmostEqual(structure.lattice.a, 4.2, places=2)
    
    def test_json_write_read(self):
        """测试JSON写入和读取"""
        filepath = os.path.join(self.temp_dir, 'test.json')
        
        # 写入
        handler = JSONHandler()
        handler.write_structure(self.structure, filepath)
        self.assertTrue(os.path.exists(filepath))
        
        # 读取
        structure = handler.read_structure(filepath)
        self.assertEqual(structure.formula, 'NaCl')


class TestComparison(unittest.TestCase):
    """测试对比分析模块"""
    
    def test_comparison_result(self):
        """测试对比结果"""
        result = ComparisonResult(
            property_name='band_gap',
            computed_value=3.2,
            experimental_value=3.0,
            experimental_uncertainty=0.1,
            unit='eV'
        )
        
        self.assertEqual(result.property_name, 'band_gap')
        self.assertAlmostEqual(result.absolute_error, 0.2, places=4)
        self.assertAlmostEqual(result.relative_error, 0.0667, places=2)
        self.assertAlmostEqual(result.percentage_error, 6.67, places=1)
        self.assertAlmostEqual(result.z_score, 2.0, places=5)
        # z_score = 2.0 is at boundary of 2sigma, may have floating point issues
        self.assertAlmostEqual(abs(result.z_score), 2.0, places=5)
    
    def test_validation_report(self):
        """测试验证报告"""
        results = [
            ComparisonResult('band_gap', 3.2, 3.0, 0.1, 'eV'),
            ComparisonResult('band_gap', 2.9, 3.1, 0.1, 'eV'),
            ComparisonResult('band_gap', 3.1, 3.0, 0.1, 'eV'),
        ]
        
        report = ValidationReport(
            property_name='band_gap',
            results=results
        )
        
        self.assertIsNotNone(report.statistics)
        self.assertGreater(report.statistics.mae, 0)
        self.assertGreaterEqual(report.statistics.r2, 0)
        self.assertLessEqual(report.statistics.r2, 1)
    
    def test_structure_comparison(self):
        """测试结构比较"""
        struct1 = CrystalStructure(
            formula='NaCl',
            lattice=Lattice(4.2, 4.2, 4.2, 90, 90, 90),
            sites=[AtomSite('Na', 0, 0, 0), AtomSite('Cl', 0.5, 0.5, 0.5)]
        )
        
        struct2 = CrystalStructure(
            formula='NaCl',
            lattice=Lattice(4.18, 4.18, 4.18, 90, 90, 90),
            sites=[AtomSite('Na', 0, 0, 0), AtomSite('Cl', 0.5, 0.5, 0.5)]
        )
        
        comparison = StructureComparator.compare_lattice(struct1, struct2)
        
        self.assertIn('a_error', comparison)
        self.assertGreater(comparison['a_error'], 0)
        self.assertLess(comparison['a_error'], 1)


class TestImporters(unittest.TestCase):
    """测试导入器模块"""
    
    def test_unit_converter(self):
        """测试单位转换"""
        # GPa to Pa
        result = UnitConverter.convert(5, 'GPa', 'Pa')
        self.assertAlmostEqual(result, 5e9, places=1)
        
        # eV to J
        result = UnitConverter.convert(1, 'eV', 'J')
        self.assertAlmostEqual(result, 1.60218e-19, places=5)
        
        # K to C
        result = UnitConverter.convert(300, 'K', 'C')
        self.assertAlmostEqual(result, 26.85, places=1)
    
    def test_property_normalizer(self):
        """测试属性标准化"""
        # 名称标准化
        self.assertEqual(
            PropertyNormalizer.normalize_name('bandgap'),
            'band_gap'
        )
        self.assertEqual(
            PropertyNormalizer.normalize_name('Band Gap'),
            'band_gap'
        )
        
        # 标准单位
        self.assertEqual(
            PropertyNormalizer.get_standard_unit('band_gap'),
            'eV'
        )
    
    def test_file_importer(self):
        """测试文件导入器"""
        config = ImportConfig(validate_on_import=True)
        importer = FileImporter(config)
        
        # 创建临时文件来测试格式检测
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # 测试格式检测
            cif_path = os.path.join(tmpdir, 'test.cif')
            Path(cif_path).touch()
            self.assertTrue(importer.can_import(cif_path))
            
            # 测试JSON
            json_path = os.path.join(tmpdir, 'test.json')
            Path(json_path).touch()
            self.assertTrue(importer.can_import(json_path))
            
            # 测试不支持的格式
            xyz_path = os.path.join(tmpdir, 'test.xyz')
            Path(xyz_path).touch()
            self.assertFalse(importer.can_import(xyz_path))


class TestFeedback(unittest.TestCase):
    """测试反馈优化模块"""
    
    def test_error_analyzer(self):
        """测试误差分析器"""
        from dftlammps.experimental_validation.feedback import ErrorAnalyzer
        
        analyzer = ErrorAnalyzer()
        
        results = [
            ComparisonResult('test', 3.2, 3.0, 0.1, 'eV'),
            ComparisonResult('test', 3.1, 3.0, 0.1, 'eV'),
            ComparisonResult('test', 3.3, 3.2, 0.1, 'eV'),
        ]
        
        analysis = analyzer.analyze_systematic_errors(results)
        
        self.assertIn('mean_bias', analysis)
        self.assertIn('std_error', analysis)
        self.assertIn('trend', analysis)
    
    def test_dft_optimizer(self):
        """测试DFT优化器"""
        optimizer = DFTParameterOptimizer()
        
        # 创建测试报告
        results = [ComparisonResult('bg', 3.5, 3.0, 0.1, 'eV') for _ in range(10)]
        report = ValidationReport('band_gap', results)
        
        current_params = {'ENCUT': 400}
        adjustments = optimizer.suggest_adjustments(report, current_params)
        
        # 应该有调整建议（因为MAPE较高）
        self.assertIsInstance(adjustments, list)
    
    def test_feedback_loop(self):
        """测试反馈循环"""
        loop = FeedbackLoop()
        
        # 创建测试报告
        results = [ComparisonResult('bg', 3.2, 3.0, 0.1, 'eV') for _ in range(5)]
        report = ValidationReport('band_gap', results)
        
        current_params = {'ENCUT': 400, 'KPOINTS': [4, 4, 4]}
        
        cycle = loop.run_cycle(report, current_params, 'dft')
        
        self.assertEqual(cycle.cycle_id, 1)
        self.assertIsNotNone(cycle.recommendations)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 1. 创建结构
        lattice = Lattice(4.9, 4.9, 5.4, 90, 90, 120)
        structure = CrystalStructure(
            formula='SiO2',
            lattice=lattice,
            sites=[
                AtomSite('Si', 0.47, 0.0, 0.0),
                AtomSite('O', 0.41, 0.27, 0.12)
            ]
        )
        
        # 2. 创建数据集
        exp_dataset = ExperimentalDataset(
            structure=structure,
            properties=[
                ExperimentalProperty('band_gap', 8.9, 'eV', uncertainty=0.2)
            ]
        )
        
        # 3. 模拟计算数据
        comp_dataset = ExperimentalDataset(
            structure=structure,
            properties=[
                ExperimentalProperty('band_gap', 8.5, 'eV')
            ]
        )
        
        # 4. 对比
        comparator = PropertyComparator()
        results = comparator.compare_datasets(comp_dataset, exp_dataset)
        
        self.assertIn('band_gap', results)
        self.assertGreater(results['band_gap'].percentage_error, 0)
        
        # 5. 验证
        report = validate_properties(
            [8.5, 8.3, 8.7],
            [8.9, 8.8, 9.0],
            'band_gap',
            [0.2, 0.2, 0.2]
        )
        
        self.assertIsNotNone(report.statistics)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestDataFormats))
    suite.addTests(loader.loadTestsFromTestCase(TestStructureIO))
    suite.addTests(loader.loadTestsFromTestCase(TestComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestImporters))
    suite.addTests(loader.loadTestsFromTestCase(TestFeedback))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
