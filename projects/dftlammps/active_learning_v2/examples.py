#!/usr/bin/env python3
"""
主动学习V2使用示例

展示如何使用active_learning_v2模块的各种功能。
"""

import numpy as np
from ase import Atoms

# 导入主动学习V2模块
from active_learning_v2 import (
    # 策略
    BayesianOptimizationStrategy,
    DPPDiversityStrategy,
    MultiFidelityStrategy,
    EvidentialLearningStrategy,
    AdaptiveHybridStrategy,
    # 不确定性量化
    EnsembleUncertainty,
    MCDropoutUncertainty,
    EvidentialUncertainty,
    # 自适应组件
    AdaptiveSampler,
    PerformanceMonitor,
    # 集成
    create_active_learning_workflow,
)


def example_1_basic_uncertainty():
    """示例1: 基本不确定性量化"""
    print("\n" + "="*60)
    print("示例1: 不确定性量化")
    print("="*60)
    
    # 生成合成数据
    np.random.seed(42)
    X_train = np.random.randn(50, 5)
    y_train = np.random.randn(50)
    X_test = np.random.randn(20, 5)
    
    # 使用集成不确定性
    ensemble_uq = EnsembleUncertainty(n_models=5, random_state=42)
    ensemble_uq.fit(X_train, y_train)
    result = ensemble_uq.quantify(X_test)
    
    print(f"Total uncertainty shape: {result.total_uncertainty.shape}")
    print(f"Mean uncertainty: {np.mean(result.total_uncertainty):.4f}")
    print(f"Top 5 uncertain samples: {result.get_top_uncertain_indices(5)}")
    
    # 使用贝叶斯GP不确定性
    gp_uq = BayesianGPUncertainty(random_state=42)
    gp_uq.fit(X_train, y_train)
    result = gp_uq.quantify(X_test)
    
    print(f"\nGP Mean uncertainty: {np.mean(result.total_uncertainty):.4f}")


def example_2_single_strategy():
    """示例2: 使用单一主动学习策略"""
    print("\n" + "="*60)
    print("示例2: 单一策略 - DPP多样性策略")
    print("="*60)
    
    from active_learning_v2.strategies import StrategyConfig
    
    # 准备数据
    np.random.seed(42)
    X_unlabeled = np.random.randn(100, 5)
    X_labeled = np.random.randn(20, 5)
    y_labeled = np.random.randn(20)
    
    # 创建DPP策略
    config = StrategyConfig(batch_size=10, verbose=True)
    strategy = DPPDiversityStrategy(
        config=config,
        quality_fn='uncertainty',
        diversity_weight=1.5
    )
    
    # 选择样本
    result = strategy.select(X_unlabeled, X_labeled, y_labeled)
    
    print(f"Selected {len(result)} samples")
    print(f"Selected indices: {result.selected_indices}")
    print(f"Selection scores: {result.selected_scores[:5]}")


def example_3_bayesian_optimization():
    """示例3: 贝叶斯优化策略"""
    print("\n" + "="*60)
    print("示例3: 贝叶斯优化策略")
    print("="*60)
    
    from active_learning_v2.strategies import StrategyConfig
    
    # 准备数据
    np.random.seed(42)
    X_unlabeled = np.random.randn(100, 5)
    X_labeled = np.random.randn(20, 5)
    y_labeled = np.random.randn(20)
    
    # 创建贝叶斯优化策略
    config = StrategyConfig(batch_size=5)
    strategy = BayesianOptimizationStrategy(
        config=config,
        acquisition='ucb',  # UCB采集函数
        beta_ucb=2.0
    )
    
    # 选择样本
    result = strategy.select(X_unlabeled, X_labeled, y_labeled)
    
    print(f"Selected {len(result)} samples")
    print(f"Acquisition values: {result.acquisition_values[:5] if result.acquisition_values is not None else 'N/A'}")


def example_4_adaptive_sampler():
    """示例4: 自适应采样器"""
    print("\n" + "="*60)
    print("示例4: 自适应采样器")
    print("="*60)
    
    # 创建自适应采样器
    sampler = AdaptiveSampler(
        default_batch_size=5,
        adaptation_frequency=3,
        verbose=True
    )
    
    # 模拟主动学习循环
    np.random.seed(42)
    X_unlabeled = np.random.randn(100, 5)
    X_labeled = np.random.randn(10, 5)
    y_labeled = np.random.randn(10)
    
    for iteration in range(5):
        print(f"\n--- Iteration {iteration} ---")
        
        # 采样
        selected_indices, metadata = sampler.sample(
            X_unlabeled, X_labeled, y_labeled
        )
        
        print(f"Strategy: {metadata['strategy']}")
        print(f"Selected: {selected_indices[:5]}")
        
        # 模拟标注 (实际应用中应该进行DFT计算)
        new_X = X_unlabeled[selected_indices]
        new_y = np.random.randn(len(selected_indices))
        
        # 更新标注数据
        X_labeled = np.vstack([X_labeled, new_X])
        y_labeled = np.concatenate([y_labeled, new_y])
        
        # 从未标注池中移除
        mask = np.ones(len(X_unlabeled), dtype=bool)
        mask[selected_indices] = False
        X_unlabeled = X_unlabeled[mask]
        
        # 记录性能
        sampler.record_performance(
            iteration=iteration,
            val_loss=1.0 / (iteration + 1),
            mean_uncertainty=0.5 / (iteration + 1),
            dft_calls=len(selected_indices),
            total_cost=len(X_labeled) * 100.0
        )
        
        # 检查状态
        status = sampler.get_status()
        print(f"Current phase: {status['current_phase']}")
    
    # 打印最终统计
    print("\n--- Final Status ---")
    status = sampler.get_status()
    print(f"Strategy ranking: {status['strategy_ranking']}")


def example_5_multi_fidelity():
    """示例5: 多保真度策略"""
    print("\n" + "="*60)
    print("示例5: 多保真度策略")
    print("="*60)
    
    from active_learning_v2.strategies import StrategyConfig
    
    # 定义保真度级别
    fidelity_levels = [
        {'name': 'classical_ff', 'method': 'Lennard-Jones'},
        {'name': 'ml_potential', 'method': 'GAP'},
        {'name': 'dft_pbe', 'method': 'DFT-PBE'},
        {'name': 'dft_hse', 'method': 'DFT-HSE06'},
    ]
    fidelity_costs = [1.0, 10.0, 1000.0, 5000.0]
    
    # 创建策略
    config = StrategyConfig(batch_size=10)
    strategy = MultiFidelityStrategy(
        config=config,
        fidelity_levels=fidelity_levels,
        fidelity_costs=fidelity_costs,
        strategy='adaptive_threshold'
    )
    
    # 准备数据
    np.random.seed(42)
    X_unlabeled = np.random.randn(100, 5)
    X_labeled = np.random.randn(50, 5)  # 已有较多标注数据
    y_labeled = np.random.randn(50)
    
    # 选择样本
    result = strategy.select(X_unlabeled, X_labeled, y_labeled)
    
    print(f"Selected {len(result)} samples")
    print(f"Target fidelity: {result.metadata.get('target_fidelity', 'N/A')}")
    
    # 查看每个样本的保真度
    if 'fidelity_map' in result.metadata:
        fidelity_map = result.metadata['fidelity_map']
        for idx in result.selected_indices[:5]:
            fid = strategy.get_fidelity_for_sample(idx, result)
            print(f"  Sample {idx}: fidelity level {fid} ({fidelity_levels[fid]['name']})")


def example_6_full_workflow():
    """示例6: 完整工作流"""
    print("\n" + "="*60)
    print("示例6: 完整主动学习工作流")
    print("="*60)
    
    # 创建工作流
    workflow = create_active_learning_workflow(
        potential_type='deepmd',
        dft_calculator='vasp',
        strategy_name='adaptive',
        work_dir='./example_workflow'
    )
    
    # 创建虚拟结构
    initial_structures = [
        Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        Atoms('H2O', positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]]),
        Atoms('H2O', positions=[[0.1, 0, 0], [1.1, 0, 0], [0.1, 1, 0]]),
    ]
    
    print(f"Created {len(initial_structures)} initial structures")
    
    # 初始化 (会执行初始DFT计算)
    print("Initializing workflow...")
    workflow.initialize(initial_structures)
    
    print(f"Labeled structures: {len(workflow.labeled_structures)}")
    print(f"Total cost: {workflow.total_cost}")
    
    # 运行几个迭代
    for i in range(3):
        print(f"\n--- Running iteration {i} ---")
        result = workflow.run_iteration()
        print(f"Selected: {result['n_selected']} structures")
        print(f"Total DFT calls: {result['total_dft_calls']}")
        print(f"Strategy used: {result['selection_strategy']}")
    
    # 获取摘要
    summary = workflow.get_summary()
    print("\n--- Workflow Summary ---")
    print(f"Total iterations: {summary['total_iterations']}")
    print(f"Total structures: {summary['total_structures']}")
    print(f"Total DFT calls: {summary['total_dft_calls']}")
    print(f"Total cost: {summary['total_cost']}")
    
    # 保存结果
    workflow.save_results()
    print("\nResults saved!")


def example_7_compare_uncertainty_methods():
    """示例7: 比较不确定性量化方法"""
    print("\n" + "="*60)
    print("示例7: 比较不确定性量化方法")
    print("="*60)
    
    from active_learning_v2.uncertainty import compare_uncertainty_methods
    
    # 生成数据
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randn(100)
    X_test = np.random.randn(50, 5)
    y_test = np.random.randn(50)
    
    # 比较不同方法
    results = compare_uncertainty_methods(
        X_train, y_train,
        X_test, y_test,
        methods=['ensemble', 'bayesian_gp']  # 简化，只测试两种方法
    )
    
    print("\nComparison Results:")
    for method, metrics in results.items():
        print(f"\n{method}:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v:.4f}")
            else:
                print(f"  {key}: {value}")


def example_8_custom_strategy():
    """示例8: 创建自定义策略"""
    print("\n" + "="*60)
    print("示例8: 自定义策略")
    print("="*60)
    
    from active_learning_v2.strategies import (
        ActiveLearningStrategy,
        StrategyConfig,
        SelectionResult
    )
    
    class CustomStrategy(ActiveLearningStrategy):
        """自定义策略：选择距离训练集最远的样本"""
        
        def select(self, X_unlabeled, X_labeled=None, y_labeled=None, model=None, **kwargs):
            # 计算到训练集的最小距离
            if X_labeled is not None and len(X_labeled) > 0:
                from scipy.spatial.distance import cdist
                distances = cdist(X_unlabeled, X_labeled)
                min_distances = np.min(distances, axis=1)
            else:
                min_distances = np.random.random(len(X_unlabeled))
            
            # 选择距离最远的样本
            selected_indices = np.argsort(min_distances)[-self.config.batch_size:][::-1]
            
            return SelectionResult(
                selected_indices=selected_indices,
                selected_scores=min_distances[selected_indices]
            )
    
    # 使用自定义策略
    np.random.seed(42)
    X_unlabeled = np.random.randn(100, 5)
    X_labeled = np.random.randn(20, 5)
    y_labeled = np.random.randn(20)
    
    strategy = CustomStrategy(StrategyConfig(batch_size=10))
    result = strategy.select(X_unlabeled, X_labeled, y_labeled)
    
    print(f"Custom strategy selected: {result.selected_indices}")
    print(f"Scores: {result.selected_scores[:5]}")


def main():
    """运行所有示例"""
    print("="*60)
    print("Active Learning V2 - 使用示例")
    print("="*60)
    
    try:
        example_1_basic_uncertainty()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_single_strategy()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_bayesian_optimization()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_4_adaptive_sampler()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    try:
        example_5_multi_fidelity()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    try:
        example_6_full_workflow()
    except Exception as e:
        print(f"Example 6 failed: {e}")
    
    try:
        example_7_compare_uncertainty_methods()
    except Exception as e:
        print(f"Example 7 failed: {e}")
    
    try:
        example_8_custom_strategy()
    except Exception as e:
        print(f"Example 8 failed: {e}")
    
    print("\n" + "="*60)
    print("所有示例完成!")
    print("="*60)


if __name__ == '__main__':
    main()
