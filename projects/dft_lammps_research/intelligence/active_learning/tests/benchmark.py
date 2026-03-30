#!/usr/bin/env python3
"""
对比实验模块 - Benchmark

实现5种先进主动学习策略与基线方法的对比实验。
生成性能对比图表和统计报告。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """实验结果"""
    strategy_name: str
    iterations: List[int] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    uncertainties: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    n_samples: List[int] = field(default_factory=list)
    training_times: List[float] = field(default_factory=list)
    
    def add_point(self, iteration: int, accuracy: float, uncertainty: float, 
                  cost: float, n_sample: int, training_time: float):
        """添加数据点"""
        self.iterations.append(iteration)
        self.accuracies.append(accuracy)
        self.uncertainties.append(uncertainty)
        self.costs.append(cost)
        self.n_samples.append(n_sample)
        self.training_times.append(training_time)
    
    def get_final_accuracy(self) -> float:
        """获取最终准确率"""
        return self.accuracies[-1] if self.accuracies else 0.0
    
    def get_total_cost(self) -> float:
        """获取总成本"""
        return self.costs[-1] if self.costs else 0.0
    
    def get_area_under_curve(self) -> float:
        """获取准确率曲线下面积 (AUC)"""
        if len(self.accuracies) < 2:
            return 0.0
        try:
            return np.trapezoid(self.accuracies, self.n_samples)
        except AttributeError:
            # Fallback for older numpy versions
            try:
                return np.trapz(self.accuracies, self.n_samples)
            except AttributeError:
                # Manual trapezoidal integration
                return np.sum((self.accuracies[:-1] + self.accuracies[1:]) / 2 * np.diff(self.n_samples))


class BenchmarkDataset:
    """
    基准数据集
    
    为对比实验生成合成数据或使用真实数据。
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        noise_level: float = 0.1,
        problem_type: str = 'regression'
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_level = noise_level
        self.problem_type = problem_type
        
        self.X, self.y = self._generate_data()
        
        # 划分训练集和测试集
        self.n_train_initial = 20
        indices = np.random.permutation(n_samples)
        
        self.train_indices = indices[self.n_train_initial:].tolist()
        self.labeled_indices = indices[:self.n_train_initial].tolist()
        self.test_indices = indices[:200].tolist()  # 前200个作为测试集
    
    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成合成数据"""
        np.random.seed(42)
        
        X = np.random.randn(self.n_samples, self.n_features)
        
        # 非线性目标函数
        if self.problem_type == 'regression':
            y = (
                np.sin(X[:, 0]) * np.cos(X[:, 1]) +
                0.5 * X[:, 2] ** 2 +
                np.exp(-X[:, 3] ** 2) +
                self.noise_level * np.random.randn(self.n_samples)
            )
        else:
            # 分类问题
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        return X, y
    
    def get_labeled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取已标注数据"""
        return self.X[self.labeled_indices], self.y[self.labeled_indices]
    
    def get_unlabeled_data(self) -> np.ndarray:
        """获取未标注数据"""
        return self.X[self.train_indices]
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取测试数据"""
        return self.X[self.test_indices], self.y[self.test_indices]
    
    def label_samples(self, indices_in_unlabeled: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """标注选中的样本"""
        # 将unlabeled中的索引转换为全局索引
        global_indices = [self.train_indices[i] for i in indices_in_unlabeled]
        
        # 从未标注池中移除
        for idx in sorted(indices_in_unlabeled, reverse=True):
            self.labeled_indices.append(self.train_indices.pop(idx))
        
        return self.X[global_indices], self.y[global_indices]


class ActiveLearningBenchmark:
    """
    主动学习基准测试
    
    对比不同主动学习策略的性能。
    """
    
    def __init__(
        self,
        dataset: Optional[BenchmarkDataset] = None,
        batch_size: int = 5,
        max_iterations: int = 50,
        random_state: int = 42
    ):
        self.dataset = dataset or BenchmarkDataset()
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        self.results: Dict[str, ExperimentResult] = {}
    
    def run_comparison(
        self,
        strategies: Optional[List[str]] = None
    ) -> Dict[str, ExperimentResult]:
        """
        运行策略对比实验
        
        Args:
            strategies: 要测试的策略列表
            
        Returns:
            各策略的实验结果
        """
        if strategies is None:
            strategies = [
                'random',
                'uncertainty',
                'bayesian_optimization',
                'dpp_diversity',
                'evidential_learning',
                'adaptive_hybrid'
            ]
        
        logger.info(f"Starting benchmark with strategies: {strategies}")
        
        for strategy_name in strategies:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing strategy: {strategy_name}")
            logger.info(f"{'='*60}")
            
            result = self._run_single_strategy(strategy_name)
            self.results[strategy_name] = result
            
            logger.info(f"Strategy {strategy_name} completed:")
            logger.info(f"  Final accuracy: {result.get_final_accuracy():.4f}")
            logger.info(f"  Total cost: {result.get_total_cost():.2f}")
            logger.info(f"  AUC: {result.get_area_under_curve():.4f}")
        
        return self.results
    
    def _run_single_strategy(self, strategy_name: str) -> ExperimentResult:
        """运行单个策略"""
        from sklearn.ensemble import RandomForestRegressor
        from active_learning_v2.strategies import (
            BayesianOptimizationStrategy,
            DPPDiversityStrategy,
            EvidentialLearningStrategy,
            AdaptiveHybridStrategy,
            StrategyConfig
        )
        from active_learning_v2.uncertainty import EnsembleUncertainty
        
        # 重置数据集
        self.dataset = BenchmarkDataset(
            n_samples=self.dataset.n_samples,
            n_features=self.dataset.n_features
        )
        
        # 创建策略
        config = StrategyConfig(batch_size=self.batch_size)
        
        if strategy_name == 'random':
            # 随机选择作为基线
            strategy = None
        elif strategy_name == 'uncertainty':
            strategy = BayesianOptimizationStrategy(config, acquisition='bald')
        elif strategy_name == 'bayesian_optimization':
            strategy = BayesianOptimizationStrategy(config, acquisition='ucb')
        elif strategy_name == 'dpp_diversity':
            strategy = DPPDiversityStrategy(config, quality_fn='uncertainty')
        elif strategy_name == 'evidential_learning':
            strategy = EvidentialLearningStrategy(config)
        elif strategy_name == 'adaptive_hybrid':
            strategy = AdaptiveHybridStrategy(config)
        else:
            strategy = None
        
        result = ExperimentResult(strategy_name=strategy_name)
        
        # 主动学习循环
        for iteration in range(self.max_iterations):
            # 获取数据
            X_labeled, y_labeled = self.dataset.get_labeled_data()
            X_unlabeled = self.dataset.get_unlabeled_data()
            X_test, y_test = self.dataset.get_test_data()
            
            if len(X_unlabeled) < self.batch_size:
                break
            
            # 训练模型
            start_time = time.time()
            model = RandomForestRegressor(n_estimators=10, random_state=self.random_state)
            model.fit(X_labeled, y_labeled)
            train_time = time.time() - start_time
            
            # 评估
            y_pred = model.predict(X_test)
            accuracy = 1 / (1 + np.mean((y_pred - y_test) ** 2))  # 转换为准确率
            
            # 估计不确定性 (使用预测方差)
            if hasattr(model, 'estimators_'):
                predictions = np.array([est.predict(X_test) for est in model.estimators_])
                uncertainty = np.mean(np.std(predictions, axis=0))
            else:
                uncertainty = 0.1
            
            # 计算成本 (模拟)
            cost = len(X_labeled) * 10.0
            
            # 记录结果
            result.add_point(
                iteration=iteration,
                accuracy=accuracy,
                uncertainty=uncertainty,
                cost=cost,
                n_sample=len(X_labeled),
                training_time=train_time
            )
            
            # 选择下一个batch
            if strategy is not None:
                selection_result = strategy.select(
                    X_unlabeled=X_unlabeled,
                    X_labeled=X_labeled,
                    y_labeled=y_labeled,
                    model=model
                )
                selected_indices = selection_result.selected_indices
            else:
                # 随机选择
                selected_indices = np.random.choice(
                    len(X_unlabeled),
                    min(self.batch_size, len(X_unlabeled)),
                    replace=False
                )
            
            # 标注选中的样本
            self.dataset.label_samples(selected_indices.tolist() if hasattr(selected_indices, 'tolist') else selected_indices)
        
        return result
    
    def generate_report(self, output_dir: str = './benchmark_results'):
        """生成对比报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计摘要
        summary = self._compute_summary_statistics()
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 绘制图表
        self._plot_results(output_dir)
        
        # 生成Markdown报告
        self._generate_markdown_report(output_dir)
        
        logger.info(f"Benchmark report saved to {output_dir}")
    
    def _compute_summary_statistics(self) -> Dict:
        """计算统计摘要"""
        summary = {}
        
        for name, result in self.results.items():
            summary[name] = {
                'final_accuracy': result.get_final_accuracy(),
                'total_cost': result.get_total_cost(),
                'auc': result.get_area_under_curve(),
                'total_time': sum(result.training_times),
                'n_iterations': len(result.iterations)
            }
        
        # 计算排名
        rankings = self._compute_rankings(summary)
        summary['rankings'] = rankings
        
        return summary
    
    def _compute_rankings(self, summary: Dict) -> Dict[str, List[str]]:
        """计算策略排名"""
        rankings = {}
        
        # 按最终准确率排名
        sorted_by_accuracy = sorted(
            [(name, data['final_accuracy']) for name, data in summary.items() if 'final_accuracy' in data],
            key=lambda x: x[1],
            reverse=True
        )
        rankings['by_accuracy'] = [name for name, _ in sorted_by_accuracy]
        
        # 按AUC排名
        sorted_by_auc = sorted(
            [(name, data['auc']) for name, data in summary.items() if 'auc' in data],
            key=lambda x: x[1],
            reverse=True
        )
        rankings['by_auc'] = [name for name, _ in sorted_by_auc]
        
        # 按成本效率排名 (AUC / 成本)
        sorted_by_efficiency = sorted(
            [(name, data['auc'] / (data['total_cost'] + 1)) for name, data in summary.items() if 'auc' in data and 'total_cost' in data],
            key=lambda x: x[1],
            reverse=True
        )
        rankings['by_efficiency'] = [name for name, _ in sorted_by_efficiency]
        
        return rankings
    
    def _plot_results(self, output_dir: Path):
        """绘制结果图表"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
            return
        
        # 1. 准确率vs样本数
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for (name, result), color in zip(self.results.items(), colors):
            # 准确率曲线
            axes[0, 0].plot(result.n_samples, result.accuracies, 
                           label=name, color=color, marker='o', markersize=3)
            
            # 不确定性曲线
            axes[0, 1].plot(result.n_samples, result.uncertainties,
                           label=name, color=color, marker='s', markersize=3)
            
            # 成本曲线
            axes[1, 0].plot(result.n_samples, result.costs,
                           label=name, color=color, marker='^', markersize=3)
            
            # 训练时间
            axes[1, 1].plot(result.iterations, np.cumsum(result.training_times),
                           label=name, color=color, marker='d', markersize=3)
        
        axes[0, 0].set_xlabel('Number of Samples')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs Number of Samples')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Number of Samples')
        axes[0, 1].set_ylabel('Uncertainty')
        axes[0, 1].set_title('Uncertainty vs Number of Samples')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Number of Samples')
        axes[1, 0].set_ylabel('Cumulative Cost')
        axes[1, 0].set_title('Cost vs Number of Samples')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Cumulative Training Time (s)')
        axes[1, 1].set_title('Training Time Accumulation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_plots.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. 雷达图比较
        self._plot_radar_chart(output_dir)
        
        logger.info(f"Plots saved to {output_dir}")
    
    def _plot_radar_chart(self, output_dir: Path):
        """绘制雷达图"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        summary = self._compute_summary_statistics()
        
        # 选择要比较的指标
        metrics = ['final_accuracy', 'auc', 'total_cost', 'total_time']
        metric_labels = ['Accuracy', 'AUC', 'Cost (inv)', 'Time (inv)']
        
        # 归一化数据
        strategies = list(self.results.keys())
        data = []
        
        for metric in metrics:
            values = [summary[s][metric] for s in strategies if s in summary and metric in summary[s]]
            if len(values) > 0:
                values = np.array(values)
                if metric in ['total_cost', 'total_time']:
                    # 反转成本和时间（越小越好）
                    values = 1 / (values + 1)
                # 归一化到0-1
                values = (values - values.min()) / (values.max() - values.min() + 1e-10)
                data.append(values)
        
        data = np.array(data).T
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
        
        for i, (strategy, color) in enumerate(zip(strategies, colors)):
            values = data[i].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Strategy Comparison (Radar Chart)', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'radar_chart.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(self, output_dir: Path):
        """生成Markdown报告"""
        summary = self._compute_summary_statistics()
        
        md_content = f"""# Active Learning V2 Benchmark Report

## 实验设置

- **Dataset**: Synthetic regression problem
- **Samples**: {self.dataset.n_samples}
- **Features**: {self.dataset.n_features}
- **Batch Size**: {self.batch_size}
- **Max Iterations**: {self.max_iterations}
- **Random State**: {self.random_state}

## 结果摘要

### 最终性能对比

| Strategy | Final Accuracy | Total Cost | AUC | Total Time (s) |
|----------|---------------|------------|-----|----------------|
"""
        
        for name, data in summary.items():
            if name == 'rankings':
                continue
            md_content += f"| {name} | {data['final_accuracy']:.4f} | {data['total_cost']:.2f} | {data['auc']:.4f} | {data['total_time']:.2f} |\n"
        
        md_content += """
### 策略排名

#### 按最终准确率排名
1. """ + '\n1. '.join(summary['rankings']['by_accuracy']) + """

#### 按AUC排名
1. """ + '\n1. '.join(summary['rankings']['by_auc']) + """

#### 按成本效率排名
1. """ + '\n1. '.join(summary['rankings']['by_efficiency']) + """

## 结论

"""
        
        # 自动分析
        best_accuracy = summary['rankings']['by_accuracy'][0] if summary['rankings']['by_accuracy'] else 'N/A'
        best_efficiency = summary['rankings']['by_efficiency'][0] if summary['rankings']['by_efficiency'] else 'N/A'
        
        md_content += f"""
- **最佳准确率策略**: {best_accuracy}
- **最佳效率策略**: {best_efficiency}

"""
        
        if 'adaptive_hybrid' in summary and 'bayesian_optimization' in summary:
            adaptive_auc = summary['adaptive_hybrid']['auc']
            bo_auc = summary['bayesian_optimization']['auc']
            improvement = ((adaptive_auc - bo_auc) / bo_auc * 100) if bo_auc > 0 else 0
            md_content += f"- **自适应混合策略相对贝叶斯优化的提升**: {improvement:.2f}%\n"
        
        md_content += """
## 图表

### 性能对比图
![Comparison Plots](comparison_plots.png)

### 雷达图
![Radar Chart](radar_chart.png)
"""
        
        with open(output_dir / 'report.md', 'w') as f:
            f.write(md_content)
        
        logger.info(f"Markdown report saved to {output_dir / 'report.md'}")


def run_benchmark(
    strategies: Optional[List[str]] = None,
    n_samples: int = 1000,
    batch_size: int = 5,
    max_iterations: int = 50,
    output_dir: str = './benchmark_results'
) -> Dict[str, ExperimentResult]:
    """
    运行基准测试的便捷函数
    
    Args:
        strategies: 要测试的策略
        n_samples: 数据集大小
        batch_size: 每批选择的样本数
        max_iterations: 最大迭代次数
        output_dir: 输出目录
        
    Returns:
        各策略的实验结果
    """
    # 创建数据集
    dataset = BenchmarkDataset(n_samples=n_samples, n_features=10)
    
    # 运行基准测试
    benchmark = ActiveLearningBenchmark(
        dataset=dataset,
        batch_size=batch_size,
        max_iterations=max_iterations
    )
    
    results = benchmark.run_comparison(strategies)
    
    # 生成报告
    benchmark.generate_report(output_dir)
    
    return results


if __name__ == '__main__':
    # 运行示例基准测试
    logging.basicConfig(level=logging.INFO)
    
    results = run_benchmark(
        strategies=[
            'random',
            'uncertainty',
            'bayesian_optimization',
            'dpp_diversity',
            'adaptive_hybrid'
        ],
        n_samples=500,
        batch_size=5,
        max_iterations=30,
        output_dir='./benchmark_results'
    )
    
    print("\nBenchmark completed!")
    print("Results saved to ./benchmark_results/")
