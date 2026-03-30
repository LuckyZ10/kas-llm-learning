#!/usr/bin/env python3
"""
自适应采样器模块 - Adaptive Sampler

实现根据模型性能自动调整采样策略的智能采样器。
包含策略选择器、性能监控器和自适应调度器。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import logging
import json
import time

logger = logging.getLogger(__name__)


class SamplingPhase(Enum):
    """采样阶段"""
    EXPLORATION = "exploration"      # 探索阶段
    EXPLOITATION = "exploitation"    # 利用阶段
    REFINEMENT = "refinement"        # 精细调整阶段
    CONVERGENCE = "convergence"      # 收敛阶段


@dataclass
class PerformanceMetrics:
    """性能指标"""
    iteration: int
    timestamp: float

    # 模型性能
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None

    # 不确定性指标
    mean_uncertainty: Optional[float] = None
    max_uncertainty: Optional[float] = None
    uncertainty_reduction: Optional[float] = None

    # 数据覆盖度
    coverage_score: Optional[float] = None
    diversity_score: Optional[float] = None

    # 计算成本
    dft_calls: int = 0
    total_cost: float = 0.0

    # 收敛指标
    error_improvement: Optional[float] = None
    convergence_rate: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'iteration': self.iteration,
            'timestamp': self.timestamp,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy,
            'mean_uncertainty': self.mean_uncertainty,
            'max_uncertainty': self.max_uncertainty,
            'uncertainty_reduction': self.uncertainty_reduction,
            'coverage_score': self.coverage_score,
            'diversity_score': self.diversity_score,
            'dft_calls': self.dft_calls,
            'total_cost': self.total_cost,
            'error_improvement': self.error_improvement,
            'convergence_rate': self.convergence_rate,
        }


@dataclass
class StrategyRecommendation:
    """策略推荐"""
    strategy_name: str
    confidence: float  # 0-1
    reason: str
    expected_improvement: float
    parameters: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    性能监控器

    追踪主动学习过程的各项性能指标，检测收敛情况，
    并提供早期停止和策略切换建议。
    """

    def __init__(
        self,
        window_size: int = 5,
        convergence_patience: int = 3,
        improvement_threshold: float = 0.01,
        min_iterations: int = 5,
        cost_budget: Optional[float] = None
    ):
        self.window_size = window_size
        self.convergence_patience = convergence_patience
        self.improvement_threshold = improvement_threshold
        self.min_iterations = min_iterations
        self.cost_budget = cost_budget

        self.metrics_history: deque = deque(maxlen=50)
        self.iteration = 0
        self.total_cost = 0.0
        self.stagnation_count = 0

        # 收敛检测
        self.best_performance = -np.inf
        self.best_iteration = 0

    def record(self, metrics: PerformanceMetrics):
        """记录性能指标"""
        self.iteration = metrics.iteration
        self.total_cost += metrics.total_cost
        self.metrics_history.append(metrics)

        # 检测性能改进
        current_perf = self._compute_overall_performance(metrics)
        if current_perf > self.best_performance + self.improvement_threshold:
            self.best_performance = current_perf
            self.best_iteration = self.iteration
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        logger.debug(f"Recorded metrics for iteration {self.iteration}")

    def _compute_overall_performance(self, metrics: PerformanceMetrics) -> float:
        """计算综合性能分数"""
        score = 0.0
        weights = []
        values = []

        if metrics.val_accuracy is not None:
            values.append(metrics.val_accuracy)
            weights.append(0.3)
        if metrics.uncertainty_reduction is not None:
            values.append(metrics.uncertainty_reduction)
            weights.append(0.2)
        if metrics.coverage_score is not None:
            values.append(metrics.coverage_score)
            weights.append(0.2)
        if metrics.diversity_score is not None:
            values.append(metrics.diversity_score)
            weights.append(0.15)
        if metrics.error_improvement is not None:
            values.append(1 / (1 + abs(metrics.error_improvement)))
            weights.append(0.15)

        if len(values) > 0:
            weights = np.array(weights) / np.sum(weights)
            score = np.dot(values, weights)

        return score

    def should_stop(self) -> Tuple[bool, str]:
        """
        判断是否应该停止

        Returns:
            (should_stop, reason)
        """
        # 检查最小迭代次数
        if self.iteration < self.min_iterations:
            return False, "Minimum iterations not reached"

        # 检查成本预算
        if self.cost_budget is not None and self.total_cost >= self.cost_budget:
            return True, f"Cost budget exceeded ({self.total_cost:.2f} >= {self.cost_budget:.2f})"

        # 检查收敛
        if self.stagnation_count >= self.convergence_patience:
            return True, f"Performance stagnated for {self.stagnation_count} iterations"

        # 检查不确定性收敛
        if len(self.metrics_history) >= self.window_size:
            recent = list(self.metrics_history)[-self.window_size:]
            mean_uncertainties = [m.mean_uncertainty for m in recent if m.mean_uncertainty is not None]

            if len(mean_uncertainties) >= self.window_size:
                uncertainty_trend = np.polyfit(range(len(mean_uncertainties)), mean_uncertainties, 1)[0]
                if abs(uncertainty_trend) < self.improvement_threshold * 0.1:
                    return True, "Uncertainty converged"

        return False, "Continue"

    def detect_phase(self) -> SamplingPhase:
        """检测当前采样阶段"""
        if self.iteration < 20:
            return SamplingPhase.EXPLORATION
        elif self.iteration < 100:
            if self.stagnation_count > 0:
                return SamplingPhase.REFINEMENT
            return SamplingPhase.EXPLOITATION
        else:
            if self.stagnation_count > self.convergence_patience // 2:
                return SamplingPhase.CONVERGENCE
            return SamplingPhase.REFINEMENT

    def get_trend(self, metric: str, n_points: int = 5) -> Optional[float]:
        """
        获取指标趋势

        Returns:
            趋势线斜率 (正值=上升，负值=下降)
        """
        if len(self.metrics_history) < n_points:
            return None

        recent = list(self.metrics_history)[-n_points:]
        values = [getattr(m, metric) for m in recent if getattr(m, metric) is not None]

        if len(values) < 3:
            return None

        trend = np.polyfit(range(len(values)), values, 1)[0]
        return trend

    def get_summary(self) -> Dict:
        """获取性能摘要"""
        if len(self.metrics_history) == 0:
            return {}

        recent = list(self.metrics_history)[-self.window_size:]

        return {
            'total_iterations': self.iteration,
            'total_cost': self.total_cost,
            'best_performance': self.best_performance,
            'best_iteration': self.best_iteration,
            'stagnation_count': self.stagnation_count,
            'current_phase': self.detect_phase().value,
            'recent_mean_val_loss': np.mean([m.val_loss for m in recent if m.val_loss is not None]),
            'recent_mean_uncertainty': np.mean([m.mean_uncertainty for m in recent if m.mean_uncertainty is not None]),
        }

    def plot_history(self, save_path: Optional[str] = None):
        """绘制性能历史"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        if len(self.metrics_history) < 2:
            logger.warning("Not enough data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        iterations = [m.iteration for m in self.metrics_history]

        # 损失曲线
        val_losses = [m.val_loss for m in self.metrics_history]
        if any(v is not None for v in val_losses):
            axes[0, 0].plot(iterations, val_losses, 'b-', label='Validation Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss History')
            axes[0, 0].legend()

        # 不确定性
        uncertainties = [m.mean_uncertainty for m in self.metrics_history]
        if any(v is not None for v in uncertainties):
            axes[0, 1].plot(iterations, uncertainties, 'r-', label='Mean Uncertainty')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Uncertainty')
            axes[0, 1].set_title('Uncertainty History')
            axes[0, 1].legend()

        # 成本
        costs = [m.total_cost for m in self.metrics_history]
        axes[1, 0].plot(iterations, np.cumsum(costs), 'g-', label='Cumulative Cost')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].set_title('Cost Accumulation')
        axes[1, 0].legend()

        # 覆盖率
        coverages = [m.coverage_score for m in self.metrics_history]
        if any(v is not None for v in coverages):
            axes[1, 1].plot(iterations, coverages, 'm-', label='Coverage')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Coverage Score')
            axes[1, 1].set_title('Data Coverage')
            axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


class StrategySelector:
    """
    策略选择器

    基于当前状态和性能历史，推荐最优采样策略。
    支持基于规则和基于学习的选择方法。
    """

    def __init__(
        self,
        strategy_pool: Optional[List[str]] = None,
        selection_method: str = 'rule_based',
        exploration_ratio: float = 0.2
    ):
        self.strategy_pool = strategy_pool or [
            'bayesian_optimization',
            'dpp_diversity',
            'uncertainty_sampling',
            'evidential_learning',
            'multi_fidelity'
        ]
        self.selection_method = selection_method
        self.exploration_ratio = exploration_ratio

        # 策略性能统计
        self.strategy_stats = defaultdict(lambda: {
            'usage_count': 0,
            'success_count': 0,
            'avg_improvement': 0.0,
            'total_cost': 0.0
        })

    def recommend(
        self,
        monitor: PerformanceMonitor,
        X_labeled: Optional[np.ndarray] = None,
        y_labeled: Optional[np.ndarray] = None
    ) -> StrategyRecommendation:
        """
        推荐采样策略
        """
        if self.selection_method == 'rule_based':
            return self._rule_based_recommendation(monitor, X_labeled, y_labeled)
        elif self.selection_method == 'bandit':
            return self._bandit_recommendation(monitor)
        else:
            return self._rule_based_recommendation(monitor, X_labeled, y_labeled)

    def _rule_based_recommendation(
        self,
        monitor: PerformanceMonitor,
        X_labeled: Optional[np.ndarray],
        y_labeled: Optional[np.ndarray]
    ) -> StrategyRecommendation:
        """
        基于规则的推荐

        规则:
        1. 冷启动 (n < 10): 随机探索
        2. 早期探索 (10 <= n < 50): 贝叶斯优化 + DPP
        3. 中期开发 (50 <= n < 200): 不确定性采样 + 证据学习
        4. 后期优化 (n >= 200): 多保真度 + 自适应混合
        5. 如果检测到停滞: 切换到探索策略
        """
        n_labeled = len(X_labeled) if X_labeled is not None else 0
        phase = monitor.detect_phase()

        # 检查是否停滞
        if monitor.stagnation_count > 0:
            return StrategyRecommendation(
                strategy_name='bayesian_optimization',
                confidence=0.8,
                reason=f"Performance stagnation detected ({monitor.stagnation_count} iterations), switching to exploration",
                expected_improvement=0.15,
                parameters={'acquisition': 'bald'}
            )

        # 基于阶段的推荐
        if n_labeled < 10:
            return StrategyRecommendation(
                strategy_name='random',
                confidence=0.9,
                reason="Cold start: insufficient labeled data",
                expected_improvement=0.3,
                parameters={}
            )

        elif phase == SamplingPhase.EXPLORATION:
            return StrategyRecommendation(
                strategy_name='dpp_diversity',
                confidence=0.75,
                reason="Exploration phase: prioritize diverse samples",
                expected_improvement=0.2,
                parameters={'quality_fn': 'distance', 'diversity_weight': 1.5}
            )

        elif phase == SamplingPhase.EXPLOITATION:
            # 根据不确定性趋势选择
            uncertainty_trend = monitor.get_trend('mean_uncertainty', 5)

            if uncertainty_trend is not None and uncertainty_trend > 0:
                return StrategyRecommendation(
                    strategy_name='evidential_learning',
                    confidence=0.8,
                    reason="Rising uncertainty: use evidential learning to identify epistemic uncertainty",
                    expected_improvement=0.12,
                    parameters={'use_epistemic': True}
                )
            else:
                return StrategyRecommendation(
                    strategy_name='uncertainty_sampling',
                    confidence=0.75,
                    reason="Exploitation phase: focus on high-uncertainty regions",
                    expected_improvement=0.1,
                    parameters={}
                )

        elif phase == SamplingPhase.REFINEMENT:
            return StrategyRecommendation(
                strategy_name='multi_fidelity',
                confidence=0.7,
                reason="Refinement phase: use multi-fidelity to optimize cost-efficiency",
                expected_improvement=0.08,
                parameters={'target_fidelity': 'adaptive'}
            )

        else:  # CONVERGENCE
            return StrategyRecommendation(
                strategy_name='adaptive_hybrid',
                confidence=0.6,
                reason="Convergence phase: balanced approach with multiple strategies",
                expected_improvement=0.05,
                parameters={'selection_mode': 'adaptive'}
            )

    def _bandit_recommendation(self, monitor: PerformanceMonitor) -> StrategyRecommendation:
        """
        基于多臂老虎机的推荐

        使用epsilon-greedy策略平衡探索与利用
        """
        # 简单的epsilon-greedy
        if np.random.random() < self.exploration_ratio:
            # 探索：随机选择
            strategy = np.random.choice(self.strategy_pool)
            return StrategyRecommendation(
                strategy_name=strategy,
                confidence=0.5,
                reason="Exploration: random strategy selection",
                expected_improvement=0.1
            )

        # 利用：选择历史性能最好的
        best_strategy = None
        best_score = -np.inf

        for strategy, stats in self.strategy_stats.items():
            if stats['usage_count'] > 0:
                # UCB分数
                score = stats['avg_improvement'] + 0.1 * np.sqrt(np.log(monitor.iteration + 1) / (stats['usage_count'] + 1))
                if score > best_score:
                    best_score = score
                    best_strategy = strategy

        if best_strategy is None:
            best_strategy = self.strategy_pool[0]

        return StrategyRecommendation(
            strategy_name=best_strategy,
            confidence=0.8,
            reason="Exploitation: best performing strategy based on history",
            expected_improvement=stats.get('avg_improvement', 0.1)
        )

    def record_outcome(self, strategy: str, improvement: float, cost: float):
        """记录策略效果"""
        stats = self.strategy_stats[strategy]
        stats['usage_count'] += 1
        stats['total_cost'] += cost

        if improvement > 0:
            stats['success_count'] += 1

        # 更新平均改进
        n = stats['usage_count']
        stats['avg_improvement'] = (stats['avg_improvement'] * (n - 1) + improvement) / n

    def get_strategy_ranking(self) -> List[Tuple[str, float]]:
        """获取策略排名"""
        scores = []
        for strategy, stats in self.strategy_stats.items():
            if stats['usage_count'] > 0:
                score = stats['avg_improvement'] / (stats['total_cost'] / stats['usage_count'] + 1e-10)
                scores.append((strategy, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)


class AdaptiveSampler:
    """
    自适应采样器

    核心组件，整合性能监控、策略选择和自适应调整。
    实现完整的自适应主动学习循环。
    """

    def __init__(
        self,
        strategies: Optional[Dict[str, Any]] = None,
        monitor: Optional[PerformanceMonitor] = None,
        selector: Optional[StrategySelector] = None,
        default_batch_size: int = 10,
        adaptation_frequency: int = 5,
        verbose: bool = True
    ):
        self.strategies = strategies or {}
        self.monitor = monitor or PerformanceMonitor()
        self.selector = selector or StrategySelector()
        self.default_batch_size = default_batch_size
        self.adaptation_frequency = adaptation_frequency
        self.verbose = verbose

        self.current_strategy = None
        self.current_recommendation = None
        self.iteration = 0

        # 策略工厂（延迟创建）
        self.strategy_factory = self._create_strategy_factory()

    def _create_strategy_factory(self) -> Dict[str, Callable]:
        """创建策略工厂函数"""
        # 从上层模块导入策略
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from strategies import (
            BayesianOptimizationStrategy,
            DPPDiversityStrategy,
            MultiFidelityStrategy,
            EvidentialLearningStrategy,
            AdaptiveHybridStrategy,
            StrategyConfig
        )

        return {
            'bayesian_optimization': lambda **kwargs: BayesianOptimizationStrategy(
                StrategyConfig(batch_size=kwargs.get('batch_size', self.default_batch_size))
            ),
            'dpp_diversity': lambda **kwargs: DPPDiversityStrategy(
                StrategyConfig(batch_size=kwargs.get('batch_size', self.default_batch_size))
            ),
            'multi_fidelity': lambda **kwargs: MultiFidelityStrategy(
                StrategyConfig(batch_size=kwargs.get('batch_size', self.default_batch_size))
            ),
            'evidential_learning': lambda **kwargs: EvidentialLearningStrategy(
                StrategyConfig(batch_size=kwargs.get('batch_size', self.default_batch_size))
            ),
            'adaptive_hybrid': lambda **kwargs: AdaptiveHybridStrategy(
                StrategyConfig(batch_size=kwargs.get('batch_size', self.default_batch_size))
            ),
            'uncertainty_sampling': lambda **kwargs: BayesianOptimizationStrategy(
                StrategyConfig(batch_size=kwargs.get('batch_size', self.default_batch_size)),
                acquisition='bald'
            ),
        }

    def sample(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray] = None,
        y_labeled: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        执行自适应采样

        Returns:
            (selected_indices, metadata)
        """
        # 检查是否需要调整策略
        if self.iteration % self.adaptation_frequency == 0:
            self._adapt_strategy(X_labeled, y_labeled)

        # 使用当前策略进行采样
        if self.current_strategy is None:
            self._adapt_strategy(X_labeled, y_labeled)

        # 执行采样
        result = self.current_strategy.select(
            X_unlabeled, X_labeled, y_labeled, model, **kwargs
        )

        metadata = {
            'strategy': self.current_strategy.get_name(),
            'recommendation_confidence': self.current_recommendation.confidence if self.current_recommendation else None,
            'recommendation_reason': self.current_recommendation.reason if self.current_recommendation else None,
            'iteration': self.iteration,
            'batch_size': len(result),
            **result.metadata
        }

        self.iteration += 1

        if self.verbose:
            logger.info(f"AdaptiveSampler: selected {len(result)} samples using {metadata['strategy']}")

        return result.selected_indices, metadata

    def _adapt_strategy(self, X_labeled: Optional[np.ndarray], y_labeled: Optional[np.ndarray]):
        """调整采样策略"""
        # 获取策略推荐
        recommendation = self.selector.recommend(self.monitor, X_labeled, y_labeled)
        self.current_recommendation = recommendation

        strategy_name = recommendation.strategy_name

        # 获取或创建策略实例
        if strategy_name in self.strategies:
            self.current_strategy = self.strategies[strategy_name]
        elif strategy_name in self.strategy_factory:
            self.current_strategy = self.strategy_factory[strategy_name](
                **recommendation.parameters
            )
            self.strategies[strategy_name] = self.current_strategy
        else:
            logger.warning(f"Unknown strategy {strategy_name}, using default")
            self.current_strategy = self.strategy_factory['bayesian_optimization']()

        if self.verbose:
            logger.info(f"Adapted to strategy: {strategy_name}")
            logger.info(f"Reason: {recommendation.reason}")

    def record_performance(
        self,
        metrics: Optional[PerformanceMetrics] = None,
        **kwargs
    ):
        """记录性能指标"""
        if metrics is None:
            metrics = PerformanceMetrics(
                iteration=self.iteration,
                timestamp=time.time(),
                **kwargs
            )

        self.monitor.record(metrics)

        # 更新策略统计
        if self.current_strategy is not None:
            improvement = metrics.error_improvement or 0
            cost = metrics.total_cost
            self.selector.record_outcome(
                self.current_strategy.get_name(),
                improvement,
                cost
            )

    def check_convergence(self) -> Tuple[bool, str]:
        """检查是否收敛"""
        return self.monitor.should_stop()

    def get_status(self) -> Dict:
        """获取当前状态"""
        should_stop, reason = self.monitor.should_stop()

        return {
            'iteration': self.iteration,
            'current_strategy': self.current_strategy.get_name() if self.current_strategy else None,
            'current_phase': self.monitor.detect_phase().value,
            'should_stop': should_stop,
            'stop_reason': reason,
            'performance_summary': self.monitor.get_summary(),
            'strategy_ranking': self.selector.get_strategy_ranking()
        }

    def save_state(self, filepath: str):
        """保存采样器状态"""
        state = {
            'iteration': self.iteration,
            'current_strategy_name': self.current_strategy.get_name() if self.current_strategy else None,
            'monitor_summary': self.monitor.get_summary(),
            'strategy_stats': dict(self.selector.strategy_stats),
            'strategy_ranking': self.selector.get_strategy_ranking()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"AdaptiveSampler state saved to {filepath}")

    def plot_performance(self, save_path: Optional[str] = None):
        """绘制性能历史"""
        self.monitor.plot_history(save_path)
