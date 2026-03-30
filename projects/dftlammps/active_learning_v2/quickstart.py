#!/usr/bin/env python3
"""
主动学习V2模块使用示例 - Quick Start
展示如何将主动学习集成到ML势训练工作流中
"""

import numpy as np
import sys
from pathlib import Path

# 添加到路径
sys.path.insert(0, str(Path(__file__).parent))

from active_learning_v2 import (
    # 策略
    BayesianOptimizationStrategy,
    DPPDiversityStrategy,
    MultiFidelityStrategy,
    EvidentialLearningStrategy,
    AdaptiveHybridStrategy,
    StrategyConfig,
    SelectionResult,
    # 不确定性量化
    EnsembleUncertainty,
    BayesianGPUncertainty,
    # 自适应组件
    AdaptiveSampler,
    PerformanceMonitor,
    StrategySelector,
)


def example_basic_usage():
    """示例1: 基本用法 - 选择下一个要计算的样本"""
    print("=" * 60)
    print("示例1: 基本用法")
    print("=" * 60)
    
    # 模拟数据
    np.random.seed(42)
    X_unlabeled = np.random.randn(100, 5)  # 100个未标注样本
    X_labeled = np.random.randn(20, 5)     # 20个已标注样本
    y_labeled = np.random.randn(20)        # 对应的标签（能量/力）
    
    # 创建策略 - 使用DPP多样性策略
    config = StrategyConfig(batch_size=5, verbose=True)
    strategy = DPPDiversityStrategy(
        config=config,
        quality_fn='distance',  # 基于距离的质量函数
        diversity_weight=1.5    # 多样性权重
    )
    
    # 选择样本
    result = strategy.select(
        X_unlabeled=X_unlabeled,
        X_labeled=X_labeled,
        y_labeled=y_labeled
    )
    
    print(f"选中的样本索引: {result.selected_indices}")
    print(f"选择分数: {result.selected_scores}")
    print(f"多样性分数: {result.diversity_scores}")
    print()
    
    return result


def example_bayesian_optimization():
    """示例2: 贝叶斯优化 - 用于昂贵的DFT计算"""
    print("=" * 60)
    print("示例2: 贝叶斯优化")
    print("=" * 60)
    
    np.random.seed(42)
    X_unlabeled = np.random.randn(100, 5)
    X_labeled = np.random.randn(20, 5)
    y_labeled = np.random.randn(20)
    
    # 创建贝叶斯优化策略
    config = StrategyConfig(batch_size=5)
    strategy = BayesianOptimizationStrategy(
        config=config,
        acquisition='ucb',  # Upper Confidence Bound
        beta_ucb=2.0        # 探索-利用权衡参数
    )
    
    result = strategy.select(X_unlabeled, X_labeled, y_labeled)
    
    print(f"策略: 贝叶斯优化 (UCB)")
    print(f"选中的样本: {result.selected_indices}")
    print(f"采集函数值: {result.acquisition_values}")
    print()


def example_adaptive_workflow():
    """示例3: 自适应工作流 - 自动选择最优策略"""
    print("=" * 60)
    print("示例3: 自适应采样工作流")
    print("=" * 60)
    
    # 创建自适应采样器
    sampler = AdaptiveSampler(
        default_batch_size=5,
        adaptation_frequency=3,  # 每3次迭代调整策略
        verbose=True
    )
    
    np.random.seed(42)
    X_unlabeled = np.random.randn(100, 5)
    X_labeled = np.random.randn(10, 5)
    y_labeled = np.random.randn(10)
    
    # 模拟主动学习循环
    for iteration in range(5):
        print(f"\n--- 迭代 {iteration} ---")
        
        # 采样
        selected_indices, metadata = sampler.sample(
            X_unlabeled, X_labeled, y_labeled
        )
        
        print(f"使用的策略: {metadata['strategy']}")
        print(f"推荐原因: {metadata.get('recommendation_reason', 'N/A')}")
        print(f"选中的样本: {selected_indices}")
        
        # 模拟DFT计算（实际应用中这里调用DFT）
        new_X = X_unlabeled[selected_indices]
        new_y = np.random.randn(len(selected_indices))
        
        # 更新训练集
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
    
    # 打印最终状态
    status = sampler.get_status()
    print("\n--- 最终状态 ---")
    print(f"当前阶段: {status['current_phase']}")
    print(f"策略排名: {status['strategy_ranking']}")
    print()


def example_uncertainty_quantification():
    """示例4: 不确定性量化 - 评估模型置信度"""
    print("=" * 60)
    print("示例4: 不确定性量化")
    print("=" * 60)
    
    np.random.seed(42)
    X_train = np.random.randn(50, 5)
    y_train = np.random.randn(50)
    X_test = np.random.randn(20, 5)
    
    # 方法1: 集成方法
    print("方法1: 集成不确定性 (Query by Committee)")
    ensemble = EnsembleUncertainty(n_models=5, random_state=42)
    ensemble.fit(X_train, y_train)
    result = ensemble.quantify(X_test)
    
    print(f"  总不确定性均值: {np.mean(result.total_uncertainty):.4f}")
    print(f"  最不确定的样本: {result.get_top_uncertain_indices(3)}")
    
    # 方法2: 贝叶斯GP
    print("\n方法2: 贝叶斯高斯过程")
    gp = BayesianGPUncertainty(random_state=42)
    gp.fit(X_train, y_train)
    result = gp.quantify(X_test)
    
    print(f"  总不确定性均值: {np.mean(result.total_uncertainty):.4f}")
    print(f"  认知不确定性: {np.mean(result.epistemic_uncertainty):.4f}")
    print(f"  偶然不确定性: {np.mean(result.aleatoric_uncertainty):.4f}")
    print()


def example_multi_fidelity():
    """示例5: 多保真度 - 结合不同精度的计算资源"""
    print("=" * 60)
    print("示例5: 多保真度主动学习")
    print("=" * 60)
    
    # 定义保真度级别（从低到高）
    fidelity_levels = [
        {'name': 'classical_ff', 'method': 'Lennard-Jones'},
        {'name': 'ml_potential', 'method': 'GAP'},
        {'name': 'dft_pbe', 'method': 'DFT-PBE'},
        {'name': 'dft_hse', 'method': 'DFT-HSE06'},
    ]
    fidelity_costs = [1.0, 10.0, 1000.0, 5000.0]
    
    config = StrategyConfig(batch_size=10)
    strategy = MultiFidelityStrategy(
        config=config,
        fidelity_levels=fidelity_levels,
        fidelity_costs=fidelity_costs,
        strategy='adaptive_threshold'
    )
    
    np.random.seed(42)
    X_unlabeled = np.random.randn(100, 5)
    X_labeled = np.random.randn(50, 5)
    y_labeled = np.random.randn(50)
    
    result = strategy.select(X_unlabeled, X_labeled, y_labeled)
    
    print(f"选中的样本数: {len(result)}")
    print(f"目标保真度级别: {result.metadata.get('target_fidelity', 'N/A')}")
    
    if 'fidelity_map' in result.metadata:
        print("\n各样本的保真度分配:")
        for idx in result.selected_indices[:5]:
            fid = strategy.get_fidelity_for_sample(idx, result)
            print(f"  样本 {idx}: {fidelity_levels[fid]['name']} (成本: {fidelity_costs[fid]})")
    print()


def example_integration():
    """示例6: 与ML势训练集成"""
    print("=" * 60)
    print("示例6: 集成到ML势训练工作流")
    print("=" * 60)
    
    try:
        from active_learning_v2.integration import (
            create_active_learning_workflow
        )
    except ImportError as e:
        print(f"跳过示例6: 需要额外依赖 - {e}")
        return
    
    # 创建工作流
    workflow = create_active_learning_workflow(
        potential_type='deepmd',
        dft_calculator='vasp',
        strategy_name='adaptive',
        work_dir='./al_workflow_example'
    )
    
    print("已创建主动学习工作流:")
    print(f"  ML势类型: deepmd")
    print(f"  DFT计算器: vasp")
    print(f"  策略: adaptive")
    print()
    
    print("使用工作流:")
    print("  1. workflow.initialize(initial_structures)")
    print("  2. workflow.run(max_iterations=50)")
    print("  3. workflow.save_results()")
    print()


def example_benchmark():
    """示例7: 运行基准测试"""
    print("=" * 60)
    print("示例7: 运行基准测试")
    print("=" * 60)
    
    from active_learning_v2.tests.benchmark import run_benchmark
    
    print("运行策略对比实验...")
    print("(这可能需要几分钟时间)")
    print()
    
    # 快速基准测试
    results = run_benchmark(
        strategies=['random', 'bayesian_optimization', 'dpp_diversity'],
        n_samples=200,
        batch_size=5,
        max_iterations=15,
        output_dir='./benchmark_example'
    )
    
    print("\n基准测试完成!")
    print("结果保存在 ./benchmark_example/")
    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("主动学习V2模块 - 使用示例")
    print("=" * 60 + "\n")
    
    examples = [
        ("基本用法", example_basic_usage),
        ("贝叶斯优化", example_bayesian_optimization),
        ("自适应工作流", example_adaptive_workflow),
        ("不确定性量化", example_uncertainty_quantification),
        ("多保真度", example_multi_fidelity),
        ("系统集成", example_integration),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"示例 '{name}' 失败: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    print("=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
