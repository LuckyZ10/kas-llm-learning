"""
closed_loop_discovery.py
闭环发现系统

整合计算、合成、表征和反馈, 实现全自动材料发现。
计算→合成→表征→反馈→优化的完整闭环。

References:
- Szymanski et al. (2021) "Toward autonomous design and synthesis of novel inorganic materials"
- 2024进展: 完全自主的材料发现系统
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict
import json


@dataclass
class ComputationResult:
    """计算结果"""
    structure_id: str
    formula: str
    predicted_properties: Dict[str, float]
    uncertainty: Dict[str, float]
    confidence: float
    computation_time: float
    method: str = "DFT"  # or "ML"


@dataclass
class SynthesisResult:
    """合成结果"""
    batch_id: str
    synthesis_params: Dict[str, Any]
    actual_yield: float
    phase_purity: float
    morphology: str
    synthesis_time: float


@dataclass
class CharacterizationResult:
    """表征结果"""
    sample_id: str
    xrd_data: Optional[np.ndarray] = None
    sem_data: Optional[np.ndarray] = None
    measured_properties: Dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.0


@dataclass
class DiscoveryIteration:
    """发现迭代"""
    iteration_id: int
    timestamp: datetime
    computation: ComputationResult
    synthesis: Optional[SynthesisResult] = None
    characterization: Optional[CharacterizationResult] = None
    feedback: Optional[Dict] = None


class BayesianOptimizer:
    """
    贝叶斯优化器
    
    用于高效采样设计空间
    """
    
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        acquisition: str = "ei"  # 'ei', 'ucb', 'pi'
    ):
        self.bounds = bounds
        self.acquisition = acquisition
        
        # 高斯过程模型 (简化)
        self.X_observed = []
        self.y_observed = []
        
    def suggest(self) -> Dict[str, float]:
        """建议下一个采样点"""
        if len(self.X_observed) < 5:
            # 随机探索
            return {
                name: np.random.uniform(low, high)
                for name, (low, high) in self.bounds.items()
            }
        
        # 基于采集函数选择
        best_score = -float('inf')
        best_point = None
        
        for _ in range(100):  # 随机采样候选
            candidate = {
                name: np.random.uniform(low, high)
                for name, (low, high) in self.bounds.items()
            }
            score = self._acquisition_function(candidate)
            
            if score > best_score:
                best_score = score
                best_point = candidate
        
        return best_point
    
    def _acquisition_function(self, x: Dict[str, float]) -> float:
        """采集函数"""
        # 简化实现: EI (Expected Improvement)
        if not self.y_observed:
            return 0.0
        
        best_y = max(self.y_observed)
        # 简化: 基于距离的不确定性估计
        distances = [
            sum((x[k] - obs[k])**2 for k in x.keys())
            for obs in self.X_observed
        ]
        min_dist = min(distances) if distances else 1.0
        uncertainty = np.exp(-min_dist)
        
        # EI近似
        return uncertainty
    
    def update(self, x: Dict[str, float], y: float):
        """更新观测"""
        self.X_observed.append(x)
        self.y_observed.append(y)


class FeedbackAnalyzer:
    """
    反馈分析器
    
    分析计算-实验差异, 指导下一轮优化
    """
    
    def __init__(self):
        self.discrepancy_history = []
        
    def analyze(
        self,
        computation: ComputationResult,
        characterization: CharacterizationResult
    ) -> Dict[str, Any]:
        """
        分析计算和实验结果的差异
        """
        feedback = {
            'discrepancies': {},
            'recommendations': [],
            'model_update_needed': False
        }
        
        # 比较预测和实测
        for prop, pred_val in computation.predicted_properties.items():
            if prop in characterization.measured_properties:
                actual_val = characterization.measured_properties[prop]
                discrepancy = abs(pred_val - actual_val) / (abs(actual_val) + 1e-8)
                
                feedback['discrepancies'][prop] = {
                    'predicted': pred_val,
                    'actual': actual_val,
                    'error': discrepancy
                }
                
                # 如果误差大, 建议更新模型
                if discrepancy > 0.2:  # 20%误差阈值
                    feedback['model_update_needed'] = True
        
        # 生成建议
        if feedback['model_update_needed']:
            feedback['recommendations'].append(
                "Retrain ML model with new experimental data"
            )
        
        if characterization.quality_score < 0.5:
            feedback['recommendations'].append(
                "Adjust synthesis conditions to improve sample quality"
            )
        
        return feedback


class ClosedLoopDiscovery:
    """
    闭环发现系统主类
    """
    
    def __init__(
        self,
        target_properties: Dict[str, Tuple[float, float]],
        max_iterations: int = 100
    ):
        self.target_properties = target_properties
        self.max_iterations = max_iterations
        
        # 组件
        self.optimizer = BayesianOptimizer({
            'composition_x': (0, 1),
            'temperature': (300, 1500),
            'time': (1, 48)
        })
        
        self.feedback_analyzer = FeedbackAnalyzer()
        
        # 状态
        self.iterations: List[DiscoveryIteration] = []
        self.current_iteration = 0
        self.best_samples: List[DiscoveryIteration] = []
        
    def run(self) -> List[DiscoveryIteration]:
        """运行闭环发现"""
        print("Starting Closed-Loop Discovery")
        print(f"Target properties: {self.target_properties}")
        
        for i in range(self.max_iterations):
            self.current_iteration = i
            print(f"\n{'='*60}")
            print(f"Iteration {i+1}/{self.max_iterations}")
            print('='*60)
            
            # 1. 计算筛选
            computation = self._computational_screening()
            
            # 2. 合成实验
            synthesis = self._synthesize(computation)
            
            # 3. 表征分析
            characterization = self._characterize(synthesis)
            
            # 4. 反馈分析
            feedback = self._analyze_feedback(computation, characterization)
            
            # 记录迭代
            iteration = DiscoveryIteration(
                iteration_id=i,
                timestamp=datetime.now(),
                computation=computation,
                synthesis=synthesis,
                characterization=characterization,
                feedback=feedback
            )
            
            self.iterations.append(iteration)
            
            # 检查是否达到目标
            if self._check_target(iteration):
                print(f"\n✓ Target achieved at iteration {i+1}!")
                self.best_samples.append(iteration)
                
                # 继续寻找更多候选, 或者停止
                if len(self.best_samples) >= 3:
                    break
            
            # 更新优化器
            score = self._score_iteration(iteration)
            self.optimizer.update(
                synthesis.synthesis_params if synthesis else {},
                score
            )
        
        return self.iterations
    
    def _computational_screening(self) -> ComputationResult:
        """计算筛选阶段"""
        print("\n[1] Computational Screening")
        
        # 获取下一个候选
        params = self.optimizer.suggest()
        
        # 模拟计算 (实际系统会调用DFT或ML模型)
        # 这里使用简化模型
        structure_id = f"struct_{self.current_iteration:04d}"
        
        # 基于参数预测性质
        predicted_props = {
            'band_gap': 1.5 + np.random.randn() * 0.3,
            'conductivity': 100 + np.random.randn() * 50,
            'stability': 0.8 + np.random.rand() * 0.2
        }
        
        uncertainty = {k: 0.1 for k in predicted_props.keys()}
        
        result = ComputationResult(
            structure_id=structure_id,
            formula="A_x B_1-x O",
            predicted_properties=predicted_props,
            uncertainty=uncertainty,
            confidence=0.7,
            computation_time=3600.0,
            method="ML"
        )
        
        print(f"  Structure: {result.structure_id}")
        print(f"  Predicted: {predicted_props}")
        
        return result
    
    def _synthesize(self, computation: ComputationResult) -> SynthesisResult:
        """合成阶段"""
        print("\n[2] Synthesis")
        
        # 基于计算结果设计合成路线
        synthesis_params = {
            'temperature': 800 + np.random.randint(-100, 100),
            'time': 12 + np.random.randint(-6, 6),
            'atmosphere': 'air',
            'precursors': ['A2O3', 'BO', 'O2']
        }
        
        # 模拟合成结果
        batch_id = f"batch_{self.current_iteration:04d}"
        
        result = SynthesisResult(
            batch_id=batch_id,
            synthesis_params=synthesis_params,
            actual_yield=0.6 + np.random.rand() * 0.3,
            phase_purity=0.7 + np.random.rand() * 0.2,
            morphology="nanoparticles",
            synthesis_time=synthesis_params['time'] * 60  # minutes
        )
        
        print(f"  Batch: {result.batch_id}")
        print(f"  Yield: {result.actual_yield:.2%}")
        print(f"  Purity: {result.phase_purity:.2%}")
        
        return result
    
    def _characterize(self, synthesis: SynthesisResult) -> CharacterizationResult:
        """表征阶段"""
        print("\n[3] Characterization")
        
        sample_id = f"sample_{self.current_iteration:04d}"
        
        # 模拟实测性质 (带有一些噪声)
        measured_props = {
            'band_gap': 1.5 + np.random.randn() * 0.2,
            'conductivity': 100 + np.random.randn() * 30,
            'stability': 0.8 + np.random.rand() * 0.15
        }
        
        # 计算质量分数
        quality = np.mean([
            1 - abs(measured_props[k] - 2.0) / 2.0  # 假设目标是2.0
            for k in measured_props.keys()
        ])
        
        result = CharacterizationResult(
            sample_id=sample_id,
            measured_properties=measured_props,
            quality_score=max(0, quality)
        )
        
        print(f"  Sample: {result.sample_id}")
        print(f"  Measured: {measured_props}")
        print(f"  Quality: {result.quality_score:.2f}")
        
        return result
    
    def _analyze_feedback(
        self,
        computation: ComputationResult,
        characterization: CharacterizationResult
    ) -> Dict:
        """反馈分析阶段"""
        print("\n[4] Feedback Analysis")
        
        feedback = self.feedback_analyzer.analyze(computation, characterization)
        
        print(f"  Discrepancies: {len(feedback['discrepancies'])}")
        if feedback['recommendations']:
            print(f"  Recommendations:")
            for rec in feedback['recommendations']:
                print(f"    - {rec}")
        
        return feedback
    
    def _check_target(self, iteration: DiscoveryIteration) -> bool:
        """检查是否达到目标"""
        if not iteration.characterization:
            return False
        
        measured = iteration.characterization.measured_properties
        
        for prop, (target, tolerance) in self.target_properties.items():
            if prop not in measured:
                return False
            if abs(measured[prop] - target) > tolerance:
                return False
        
        return True
    
    def _score_iteration(self, iteration: DiscoveryIteration) -> float:
        """评分迭代结果"""
        if not iteration.characterization:
            return 0.0
        
        score = iteration.characterization.quality_score
        
        # 考虑与目标的接近程度
        for prop, (target, tolerance) in self.target_properties.items():
            if prop in iteration.characterization.measured_properties:
                actual = iteration.characterization.measured_properties[prop]
                error = abs(actual - target) / tolerance
                score += max(0, 1 - error)
        
        return score
    
    def get_summary(self) -> Dict:
        """获取发现摘要"""
        return {
            'total_iterations': len(self.iterations),
            'successful_iterations': len(self.best_samples),
            'best_samples': [
                {
                    'iteration': it.iteration_id,
                    'properties': it.characterization.measured_properties if it.characterization else None,
                    'synthesis_params': it.synthesis.synthesis_params if it.synthesis else None
                }
                for it in self.best_samples
            ],
            'convergence_history': [
                {
                    'iteration': it.iteration_id,
                    'score': self._score_iteration(it)
                }
                for it in self.iterations
            ]
        }
    
    def export_results(self, filename: str):
        """导出结果"""
        summary = self.get_summary()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nResults exported to {filename}")


class AdaptiveModel:
    """
    自适应模型
    
    根据实验反馈更新ML模型
    """
    
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.training_data = []
        
    def update(self, new_data: List[Tuple[Dict, float]]):
        """用新数据更新模型"""
        self.training_data.extend(new_data)
        
        # 重训练或微调
        # 简化: 记录数据
        print(f"Model updated with {len(new_data)} new samples")
        print(f"Total training data: {len(self.training_data)}")
    
    def predict(self, x: Dict) -> float:
        """预测"""
        # 使用基础模型
        return 0.0  # 简化


if __name__ == "__main__":
    print("=" * 60)
    print("Closed-Loop Discovery Demo")
    print("=" * 60)
    
    # 创建闭环发现系统
    discovery = ClosedLoopDiscovery(
        target_properties={
            'band_gap': (2.0, 0.3),
            'conductivity': (150.0, 30.0),
            'stability': (0.9, 0.1)
        },
        max_iterations=5
    )
    
    # 运行发现
    iterations = discovery.run()
    
    # 输出摘要
    print("\n" + "=" * 60)
    print("Discovery Summary")
    print("=" * 60)
    
    summary = discovery.get_summary()
    print(f"\nTotal iterations: {summary['total_iterations']}")
    print(f"Successful discoveries: {summary['successful_iterations']}")
    
    if summary['best_samples']:
        print("\nBest samples found:")
        for sample in summary['best_samples']:
            print(f"  Iteration {sample['iteration']}: {sample['properties']}")
    
    # 导出结果
    discovery.export_results("discovery_results.json")
    
    print("\n" + "=" * 60)
    print("Closed-Loop Discovery Demo completed!")
    print("Key features:")
    print("- Computational screening")
    print("- Automated synthesis")
    print("- Characterization feedback")
    print("- Bayesian optimization")
    print("- Adaptive learning")
    print("=" * 60)
