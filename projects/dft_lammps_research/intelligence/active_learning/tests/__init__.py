#!/usr/bin/env python3
"""
测试模块 - Tests

为主动学习V2模块提供单元测试和集成测试。
"""

import numpy as np
import unittest
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from active_learning_v2.uncertainty import (
    EnsembleUncertainty,
    MCDropoutUncertainty,
    EvidentialUncertainty,
    BayesianGPUncertainty,
    UncertaintyResult
)

from active_learning_v2.strategies import (
    BayesianOptimizationStrategy,
    DPPDiversityStrategy,
    MultiFidelityStrategy,
    EvidentialLearningStrategy,
    AdaptiveHybridStrategy,
    StrategyConfig,
)

from active_learning_v2.adaptive import (
    AdaptiveSampler,
    PerformanceMonitor,
    StrategySelector,
    PerformanceMetrics
)


class TestUncertaintyQuantifiers(unittest.TestCase):
    """测试不确定性量化器"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 5
        
        self.X_train = np.random.randn(50, self.n_features)
        self.y_train = np.random.randn(50)
        
        self.X_test = np.random.randn(self.n_samples, self.n_features)
    
    def test_ensemble_uncertainty(self):
        """测试集成不确定性"""
        uq = EnsembleUncertainty(n_models=3, random_state=42)
        uq.fit(self.X_train, self.y_train)
        
        result = uq.quantify(self.X_test)
        
        self.assertIsInstance(result, UncertaintyResult)
        self.assertEqual(len(result.total_uncertainty), self.n_samples)
        self.assertTrue(np.all(result.total_uncertainty >= 0))
        self.assertTrue(np.all(result.confidence >= 0))
        self.assertTrue(np.all(result.confidence <= 1))
    
    def test_bayesian_gp_uncertainty(self):
        """测试贝叶斯GP不确定性"""
        uq = BayesianGPUncertainty(random_state=42)
        uq.fit(self.X_train, self.y_train)
        
        result = uq.quantify(self.X_test)
        
        self.assertIsInstance(result, UncertaintyResult)
        self.assertEqual(len(result.total_uncertainty), self.n_samples)
        self.assertTrue(np.all(result.total_uncertainty >= 0))
    
    def test_uncertainty_result(self):
        """测试不确定性结果类"""
        total = np.random.random(10)
        epistemic = np.random.random(10)
        aleatoric = np.random.random(10)
        predictions = np.random.random(10)
        confidence = np.random.random(10)
        
        result = UncertaintyResult(
            total_uncertainty=total,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            predictions=predictions,
            confidence=confidence
        )
        
        # 测试top-k选择
        top_indices = result.get_top_uncertain_indices(3)
        self.assertEqual(len(top_indices), 3)
        
        # 测试统计信息
        stats = result.get_uncertainty_stats()
        self.assertIn('mean_total', stats)
        self.assertIn('max_total', stats)


class TestActiveLearningStrategies(unittest.TestCase):
    """测试主动学习策略"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        
        self.n_unlabeled = 50
        self.n_labeled = 20
        self.n_features = 5
        self.batch_size = 5
        
        self.X_unlabeled = np.random.randn(self.n_unlabeled, self.n_features)
        self.X_labeled = np.random.randn(self.n_labeled, self.n_features)
        self.y_labeled = np.random.randn(self.n_labeled)
    
    def test_bayesian_optimization_strategy(self):
        """测试贝叶斯优化策略"""
        config = StrategyConfig(batch_size=self.batch_size)
        strategy = BayesianOptimizationStrategy(config, acquisition='ucb')
        
        result = strategy.select(
            self.X_unlabeled,
            self.X_labeled,
            self.y_labeled
        )
        
        self.assertEqual(len(result), self.batch_size)
        self.assertTrue(all(0 <= idx < self.n_unlabeled for idx in result.selected_indices))
    
    def test_dpp_diversity_strategy(self):
        """测试DPP多样性策略"""
        config = StrategyConfig(batch_size=self.batch_size)
        strategy = DPPDiversityStrategy(config, quality_fn='random')
        
        result = strategy.select(
            self.X_unlabeled,
            self.X_labeled,
            self.y_labeled
        )
        
        self.assertEqual(len(result), self.batch_size)
        self.assertEqual(len(set(result.selected_indices)), self.batch_size)
    
    def test_multi_fidelity_strategy(self):
        """测试多保真度策略"""
        config = StrategyConfig(batch_size=self.batch_size)
        strategy = MultiFidelityStrategy(config)
        
        result = strategy.select(
            self.X_unlabeled,
            self.X_labeled,
            self.y_labeled
        )
        
        self.assertEqual(len(result), self.batch_size)
        self.assertIn('target_fidelity', result.metadata)
    
    def test_evidential_learning_strategy(self):
        """测试证据学习策略"""
        config = StrategyConfig(batch_size=self.batch_size)
        strategy = EvidentialLearningStrategy(config)
        
        result = strategy.select(
            self.X_unlabeled,
            self.X_labeled,
            self.y_labeled
        )
        
        self.assertEqual(len(result), self.batch_size)
    
    def test_adaptive_hybrid_strategy(self):
        """测试自适应混合策略"""
        config = StrategyConfig(batch_size=self.batch_size)
        strategy = AdaptiveHybridStrategy(config)
        
        result = strategy.select(
            self.X_unlabeled,
            self.X_labeled,
            self.y_labeled
        )
        
        self.assertEqual(len(result), self.batch_size)


class TestAdaptiveComponents(unittest.TestCase):
    """测试自适应组件"""
    
    def test_performance_monitor(self):
        """测试性能监控器"""
        monitor = PerformanceMonitor(
            convergence_patience=10,  # 增加耐心值
            min_iterations=3,
            improvement_threshold=0.01
        )
        
        # 记录性能 - 持续改进
        for i in range(5):
            metrics = PerformanceMetrics(
                iteration=i,
                timestamp=float(i),
                mean_uncertainty=1.0 / (i + 1),  # 持续下降
                val_loss=1.0 / (i + 1)  # 持续下降
            )
            monitor.record(metrics)
        
        # 检查收敛 - 应该还没收敛
        should_stop, reason = monitor.should_stop()
        self.assertFalse(should_stop)  # 还在改进
        
        # 检查阶段
        phase = monitor.detect_phase()
        self.assertIsNotNone(phase)
        
        # 获取趋势
        trend = monitor.get_trend('mean_uncertainty', 3)
        self.assertLess(trend, 0)  # 不确定性在下降
    
    def test_strategy_selector(self):
        """测试策略选择器"""
        selector = StrategySelector(selection_method='rule_based')
        monitor = PerformanceMonitor()
        
        # 添加一些历史记录
        for i in range(10):
            metrics = PerformanceMetrics(
                iteration=i,
                timestamp=float(i),
                mean_uncertainty=1.0 / (i + 1)
            )
            monitor.record(metrics)
        
        X_labeled = np.random.randn(100, 5)
        y_labeled = np.random.randn(100)
        
        recommendation = selector.recommend(monitor, X_labeled, y_labeled)
        
        self.assertIsNotNone(recommendation.strategy_name)
        self.assertGreaterEqual(recommendation.confidence, 0)
        self.assertLessEqual(recommendation.confidence, 1)
    
    def test_adaptive_sampler(self):
        """测试自适应采样器"""
        sampler = AdaptiveSampler(default_batch_size=5)
        
        X_unlabeled = np.random.randn(50, 5)
        X_labeled = np.random.randn(20, 5)
        y_labeled = np.random.randn(20)
        
        # 采样
        indices, metadata = sampler.sample(X_unlabeled, X_labeled, y_labeled)
        
        self.assertEqual(len(indices), 5)
        self.assertIn('strategy', metadata)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        try:
            from active_learning_v2.integration import (
                MLPotentialTrainer,
                DFTInterface,
                ActiveLearningV2Workflow
            )
            from ase import Atoms
        except ImportError as e:
            self.skipTest(f"Required dependency not available: {e}")
        
        # 创建组件
        ml_trainer = MLPotentialTrainer(potential_type='deepmd')
        dft_interface = DFTInterface(calculator='vasp')
        
        workflow = ActiveLearningV2Workflow(
            ml_trainer=ml_trainer,
            dft_interface=dft_interface,
            work_dir='./test_al_workflow'
        )
        
        # 创建虚拟结构
        initial_structures = [
            Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            Atoms('H2O', positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]]),
        ]
        
        # 初始化
        workflow.initialize(initial_structures)
        
        self.assertGreater(len(workflow.labeled_structures), 0)
        self.assertGreater(workflow.total_cost, 0)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestUncertaintyQuantifiers))
    suite.addTests(loader.loadTestsFromTestCase(TestActiveLearningStrategies))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
