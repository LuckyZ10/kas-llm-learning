"""
Screening RL Integration
========================

高通量筛选与RL集成
- 结合生成模型与筛选流程
- 主动学习RL
- 批次生成与筛选
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ScreeningRLConfig:
    """筛选RL配置"""
    batch_size: int = 100
    top_k_selection: int = 10
    num_iterations: int = 10
    exploration_ratio: float = 0.2
    use_uncertainty: bool = True
    use_diversity: bool = True


class ScreeningRLIntegration:
    """
    筛选与RL集成
    
    将GFlowNet/RL生成的样本集成到高通量筛选流程
    """
    
    def __init__(
        self,
        generator: torch.nn.Module,
        scorer: Any,  # 可以是模拟器、ML模型或实际实验
        config: Optional[ScreeningRLConfig] = None
    ):
        self.generator = generator
        self.scorer = scorer
        self.config = config or ScreeningRLConfig()
        
        # 生成的候选库
        self.candidate_library = []
        
        # 筛选历史
        self.screening_history = []
    
    def generate_and_screen(
        self,
        env,
        num_iterations: int = None
    ) -> List[Dict[str, Any]]:
        """
        生成并筛选候选
        
        Args:
            env: 生成环境
            num_iterations: 迭代次数
            
        Returns:
            筛选后的候选列表
        """
        num_iterations = num_iterations or self.config.num_iterations
        
        print(f"Starting generate-and-screen for {num_iterations} iterations")
        
        all_candidates = []
        
        for iteration in range(num_iterations):
            # 生成批次
            candidates = self._generate_batch(env)
            
            # 评分
            scored_candidates = self._score_candidates(candidates)
            
            # 选择
            selected = self._select_candidates(scored_candidates)
            
            # 更新生成器 (如果有反馈)
            self._update_generator(selected)
            
            # 记录
            all_candidates.extend(scored_candidates)
            self.screening_history.append({
                'iteration': iteration,
                'num_generated': len(candidates),
                'num_selected': len(selected),
                'mean_score': np.mean([c['score'] for c in scored_candidates]),
            })
            
            print(f"Iter {iteration + 1}: Generated {len(candidates)}, "
                  f"Mean score: {self.screening_history[-1]['mean_score']:.3f}")
        
        # 返回所有候选，按分数排序
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        return all_candidates
    
    def _generate_batch(self, env) -> List[Dict[str, Any]]:
        """生成候选批次"""
        self.generator.eval()
        
        candidates = []
        
        with torch.no_grad():
            for _ in range(self.config.batch_size):
                # 使用生成器生成
                if hasattr(self.generator, 'generate_samples'):
                    samples = self.generator.generate_samples(env, num_samples=1)
                    if samples:
                        candidates.append(samples[0])
                else:
                    # 默认行为
                    sample = env.get_sample()
                    candidates.append({'sample': sample})
        
        return candidates
    
    def _score_candidates(self, candidates: List[Dict]) -> List[Dict[str, Any]]:
        """对候选进行评分"""
        scored = []
        
        for candidate in candidates:
            sample = candidate.get('sample', candidate)
            
            # 使用评分器
            if hasattr(self.scorer, 'score'):
                score = self.scorer.score(sample)
            elif hasattr(self.scorer, 'predict'):
                score = self.scorer.predict(sample)
            elif callable(self.scorer):
                score = self.scorer(sample)
            else:
                score = np.random.random()  # 模拟评分
            
            scored.append({
                'candidate': candidate,
                'sample': sample,
                'score': score,
            })
        
        return scored
    
    def _select_candidates(
        self,
        scored_candidates: List[Dict]
    ) -> List[Dict]:
        """选择候选"""
        if not scored_candidates:
            return []
        
        # 按分数排序
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        selected = []
        
        # 选择高分候选
        n_top = int(self.config.top_k_selection * (1 - self.config.exploration_ratio))
        selected.extend(scored_candidates[:n_top])
        
        # 探索: 随机选择一些候选
        if self.config.exploration_ratio > 0:
            n_explore = self.config.top_k_selection - n_top
            remaining = scored_candidates[n_top:]
            
            if remaining and n_explore > 0:
                # 可以使用不确定性采样
                if self.config.use_uncertainty and len(remaining) > n_explore:
                    # 选择不确定性高的候选
                    uncertainties = [np.random.random() for _ in remaining]
                    indices = np.argsort(uncertainties)[-n_explore:]
                    selected.extend([remaining[i] for i in indices])
                else:
                    # 随机选择
                    indices = np.random.choice(
                        len(remaining),
                        min(n_explore, len(remaining)),
                        replace=False
                    )
                    selected.extend([remaining[i] for i in indices])
        
        return selected
    
    def _update_generator(self, selected_candidates: List[Dict]):
        """基于筛选结果更新生成器"""
        # 这里可以实现基于反馈的训练
        # 例如，使用高分候选作为正样本进行训练
        pass
    
    def iterative_screening(
        self,
        env,
        oracle: Any,  # 高保真度评估
        surrogate: Any,  # 低保真度代理模型
        budget: int = 100
    ) -> List[Dict[str, Any]]:
        """
        迭代筛选
        
        使用代理模型进行预筛选，然后用oracle验证
        """
        results = []
        
        for iteration in range(budget):
            # 生成候选
            candidates = self._generate_batch(env)
            
            # 代理模型预筛选
            if surrogate is not None:
                proxy_scores = []
                for candidate in candidates:
                    sample = candidate.get('sample', candidate)
                    score = surrogate(sample) if callable(surrogate) else 0.5
                    proxy_scores.append(score)
                
                # 选择最有希望的
                top_indices = np.argsort(proxy_scores)[-5:]  # 前5个
                candidates = [candidates[i] for i in top_indices]
            
            # Oracle验证
            for candidate in candidates:
                sample = candidate.get('sample', candidate)
                true_score = oracle(sample) if callable(oracle) else np.random.random()
                
                results.append({
                    'candidate': candidate,
                    'score': true_score,
                    'iteration': iteration,
                })
                
                # 更新代理模型
                if surrogate is not None and hasattr(surrogate, 'update'):
                    surrogate.update(sample, true_score)
        
        # 排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results


class ActiveLearningRL:
    """
    主动学习RL
    
    结合主动学习与RL进行高效探索
    """
    
    def __init__(
        self,
        generator: torch.nn.Module,
        uncertainty_model: Any,
        reward_model: Any
    ):
        self.generator = generator
        self.uncertainty_model = uncertainty_model
        self.reward_model = reward_model
        
        # 已标记数据
        self.labeled_data = []
        self.unlabeled_pool = []
    
    def query_strategy(
        self,
        candidates: List[Dict],
        strategy: str = "uncertainty"
    ) -> List[Dict]:
        """
        查询策略
        
        Args:
            strategy: "uncertainty", "diversity", "expected_improvement"
        """
        if strategy == "uncertainty":
            return self._uncertainty_sampling(candidates)
        elif strategy == "diversity":
            return self._diversity_sampling(candidates)
        elif strategy == "expected_improvement":
            return self._ei_sampling(candidates)
        else:
            return candidates[:10]  # 默认前10
    
    def _uncertainty_sampling(self, candidates: List[Dict]) -> List[Dict]:
        """不确定性采样"""
        uncertainties = []
        
        for candidate in candidates:
            sample = candidate.get('sample', candidate)
            
            if hasattr(self.uncertainty_model, 'predict_uncertainty'):
                unc = self.uncertainty_model.predict_uncertainty(sample)
            else:
                unc = np.random.random()
            
            uncertainties.append(unc)
        
        # 选择不确定性最高的
        indices = np.argsort(uncertainties)[-10:]
        return [candidates[i] for i in indices]
    
    def _diversity_sampling(self, candidates: List[Dict]) -> List[Dict]:
        """多样性采样"""
        # 贪心选择以最大化多样性
        selected = []
        remaining = list(range(len(candidates)))
        
        while len(selected) < 10 and remaining:
            if not selected:
                # 随机选择第一个
                idx = np.random.choice(remaining)
            else:
                # 选择距离已选最远的
                max_min_dist = -1
                idx = remaining[0]
                
                for i in remaining:
                    min_dist = min(
                        self._distance(candidates[i], candidates[s])
                        for s in selected
                    )
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        idx = i
            
            selected.append(idx)
            remaining.remove(idx)
        
        return [candidates[i] for i in selected]
    
    def _ei_sampling(self, candidates: List[Dict]) -> List[Dict]:
        """期望改进采样"""
        if not self.labeled_data:
            return candidates[:10]
        
        # 获取当前最佳
        best_score = max(d['score'] for d in self.labeled_data)
        
        eis = []
        for candidate in candidates:
            sample = candidate.get('sample', candidate)
            
            # 预测均值和不确定性
            if hasattr(self.reward_model, 'predict_with_uncertainty'):
                mean, std = self.reward_model.predict_with_uncertainty(sample)
            else:
                mean = np.random.random()
                std = 0.1
            
            # 计算EI
            z = (mean - best_score) / (std + 1e-6)
            ei = (mean - best_score) * (0.5 * (1 + np.sign(z)))  # 简化EI
            eis.append(ei)
        
        indices = np.argsort(eis)[-10:]
        return [candidates[i] for i in indices]
    
    def _distance(self, c1: Dict, c2: Dict) -> float:
        """计算候选距离"""
        s1 = c1.get('sample', c1)
        s2 = c2.get('sample', c2)
        
        if isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray):
            return np.linalg.norm(s1 - s2)
        
        return 0.0 if str(s1) == str(s2) else 1.0
    
    def active_learning_loop(
        self,
        env,
        oracle: Any,
        num_iterations: int = 10,
        samples_per_iter: int = 10
    ) -> List[Dict]:
        """主动学习循环"""
        results = []
        
        for iteration in range(num_iterations):
            # 生成候选
            candidates = []
            for _ in range(samples_per_iter * 5):
                if hasattr(self.generator, 'generate_samples'):
                    samples = self.generator.generate_samples(env, num_samples=1)
                    candidates.extend(samples)
            
            # 查询策略选择
            to_label = self.query_strategy(candidates, strategy="uncertainty")
            
            # 获取标签
            for candidate in to_label:
                sample = candidate.get('sample', candidate)
                score = oracle(sample) if callable(oracle) else np.random.random()
                
                self.labeled_data.append({
                    'sample': sample,
                    'score': score,
                })
                
                results.append({
                    'sample': sample,
                    'score': score,
                    'iteration': iteration,
                })
            
            # 更新模型
            self._update_models()
            
            print(f"Iteration {iteration + 1}: {len(to_label)} samples labeled, "
                  f"Best score: {max(d['score'] for d in self.labeled_data):.3f}")
        
        return results
    
    def _update_models(self):
        """更新不确定性模型和奖励模型"""
        # 这里可以实现模型更新逻辑
        pass


def demo():
    """演示筛选RL集成"""
    print("=" * 60)
    print("Screening RL Integration Demo")
    print("=" * 60)
    
    # 创建模拟组件
    class MockGenerator:
        def generate_samples(self, env, num_samples=1):
            samples = []
            for _ in range(num_samples):
                sample = {
                    'molecule': f"C{np.random.randint(1, 10)}H{np.random.randint(1, 20)}",
                    'property': np.random.random(),
                }
                samples.append({'sample': sample})
            return samples
    
    class MockScorer:
        def score(self, sample):
            # 模拟评分
            return sample.get('property', 0) + np.random.randn() * 0.1
    
    # 创建集成
    generator = MockGenerator()
    scorer = MockScorer()
    
    integration = ScreeningRLIntegration(
        generator,
        scorer,
        config=ScreeningRLConfig(
            batch_size=20,
            top_k_selection=5,
            num_iterations=3
        )
    )
    
    # 模拟环境
    class MockEnv:
        pass
    
    env = MockEnv()
    
    # 运行生成和筛选
    print("\nRunning generate-and-screen...")
    results = integration.generate_and_screen(env)
    
    print(f"\nTop 5 candidates:")
    for i, result in enumerate(results[:5]):
        print(f"  {i+1}. Score: {result['score']:.3f}, Sample: {result['sample']}")
    
    # 主动学习演示
    print("\nActive Learning demo...")
    
    al_rl = ActiveLearningRL(
        generator=generator,
        uncertainty_model=None,
        reward_model=None
    )
    
    def mock_oracle(sample):
        return sample.get('property', 0) + np.random.randn() * 0.05
    
    al_results = al_rl.active_learning_loop(
        env,
        mock_oracle,
        num_iterations=3,
        samples_per_iter=5
    )
    
    print(f"Active learning collected {len(al_results)} samples")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
