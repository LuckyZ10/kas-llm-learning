"""
案例推理模块 - Case-Based Reasoning for Materials Science

实现基于案例的推理系统，支持案例检索、重用、修正和学习。
特别适用于材料发现和设计任务。
"""

from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CaseStatus(Enum):
    """案例状态"""
    VERIFIED = auto()       # 已验证
    PENDING = auto()        # 待验证
    DEPRECATED = auto()     # 已废弃


@dataclass
class MaterialCase:
    """
    材料案例
    
    包含问题描述、解决方案和结果的完整材料设计案例。
    """
    case_id: str
    
    # 问题描述
    problem: Dict[str, Any] = field(default_factory=dict)
    
    # 解决方案
    solution: Dict[str, Any] = field(default_factory=dict)
    
    # 结果
    outcome: Dict[str, Any] = field(default_factory=dict)
    
    # 特征向量（用于相似度计算）
    features: Optional[np.ndarray] = None
    
    # 元数据
    status: CaseStatus = CaseStatus.VERIFIED
    success_score: float = 0.0  # 0-1之间的成功分数
    usage_count: int = 0
    timestamp: int = 0
    tags: List[str] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.case_id)
    
    def __eq__(self, other):
        if not isinstance(other, MaterialCase):
            return False
        return self.case_id == other.case_id
    
    def to_vector(self, feature_extractor: Callable = None) -> np.ndarray:
        """将案例转换为特征向量"""
        if self.features is not None:
            return self.features
        
        if feature_extractor:
            return feature_extractor(self)
        
        # 默认特征提取
        features = []
        
        # 问题特征
        if 'target_property' in self.problem:
            features.append(hash(self.problem['target_property']) % 1000)
        if 'constraints' in self.problem:
            features.append(len(self.problem['constraints']))
        
        # 解决方案特征
        if 'composition' in self.solution:
            features.append(len(self.solution['composition']))
        if 'structure_type' in self.solution:
            features.append(hash(self.solution['structure_type']) % 1000)
        
        # 结果特征
        if 'properties' in self.outcome:
            for prop, value in self.outcome['properties'].items():
                if isinstance(value, (int, float)):
                    features.append(value)
        
        return np.array(features, dtype=np.float32)


class SimilarityMetric:
    """相似度度量"""
    
    @staticmethod
    def euclidean(case1: MaterialCase, case2: MaterialCase) -> float:
        """欧氏距离相似度"""
        v1 = case1.to_vector()
        v2 = case2.to_vector()
        
        # 确保维度相同
        max_len = max(len(v1), len(v2))
        v1 = np.pad(v1, (0, max_len - len(v1)))
        v2 = np.pad(v2, (0, max_len - len(v2)))
        
        dist = np.linalg.norm(v1 - v2)
        return 1 / (1 + dist)
    
    @staticmethod
    def cosine(case1: MaterialCase, case2: MaterialCase) -> float:
        """余弦相似度"""
        v1 = case1.to_vector()
        v2 = case2.to_vector()
        
        max_len = max(len(v1), len(v2))
        v1 = np.pad(v1, (0, max_len - len(v1)))
        v2 = np.pad(v2, (0, max_len - len(v2)))
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    @staticmethod
    def weighted(case1: MaterialCase, 
                case2: MaterialCase,
                weights: Dict[str, float]) -> float:
        """加权相似度"""
        score = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            v1 = case1.problem.get(feature)
            v2 = case2.problem.get(feature)
            
            if v1 is not None and v2 is not None:
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    # 数值特征：使用归一化差值
                    diff = abs(v1 - v2) / max(abs(v1) + abs(v2), 1e-8)
                    score += weight * (1 - diff)
                elif v1 == v2:
                    # 离散特征：精确匹配
                    score += weight
                
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0


class CaseRetriever:
    """
    案例检索器
    
    支持多种检索策略和索引结构。
    """
    
    def __init__(self, 
                 similarity_func: Callable = SimilarityMetric.cosine,
                 index_type: str = "linear"):
        self.similarity_func = similarity_func
        self.index_type = index_type
        self.cases: List[MaterialCase] = []
        self.index: Any = None
        
        if index_type == "kd_tree":
            from scipy.spatial import KDTree
            self.kd_tree = None
    
    def add_case(self, case: MaterialCase):
        """添加案例"""
        self.cases.append(case)
        self._update_index()
    
    def add_cases(self, cases: List[MaterialCase]):
        """批量添加案例"""
        self.cases.extend(cases)
        self._update_index()
    
    def _update_index(self):
        """更新索引"""
        if self.index_type == "kd_tree" and len(self.cases) > 0:
            try:
                from scipy.spatial import KDTree
                vectors = np.array([c.to_vector() for c in self.cases])
                self.kd_tree = KDTree(vectors)
            except ImportError:
                pass
    
    def retrieve(self,
                query: MaterialCase,
                k: int = 5,
                min_similarity: float = 0.0) -> List[Tuple[MaterialCase, float]]:
        """
        检索最相似的k个案例
        
        Returns:
            List of (case, similarity) tuples
        """
        # 计算所有相似度
        similarities = []
        for case in self.cases:
            if case.case_id != query.case_id:  # 排除查询本身
                sim = self.similarity_func(case, query)
                if sim >= min_similarity:
                    similarities.append((case, sim))
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def retrieve_by_problem(self,
                           problem: Dict[str, Any],
                           k: int = 5) -> List[Tuple[MaterialCase, float]]:
        """根据问题描述检索"""
        query_case = MaterialCase(
            case_id="query",
            problem=problem,
            solution={},
            outcome={}
        )
        return self.retrieve(query_case, k)
    
    def retrieve_by_features(self,
                            features: np.ndarray,
                            k: int = 5) -> List[Tuple[MaterialCase, float]]:
        """根据特征向量检索"""
        query_case = MaterialCase(
            case_id="query",
            features=features
        )
        return self.retrieve(query_case, k)


class NeuralCaseEncoder(nn.Module):
    """
    神经案例编码器
    
    学习案例的深层嵌入表示。
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 embedding_dim: int = 64):
        super().__init__()
        
        # 构建编码器网络
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)
        
        # 解码器（用于重建）
        decoder_layers = []
        prev_dim = embedding_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """编码为嵌入向量"""
        return self.encoder(features)
    
    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """从嵌入向量解码"""
        return self.decoder(embedding)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        embedding = self.encode(features)
        reconstructed = self.decode(embedding)
        return embedding, reconstructed
    
    def compute_similarity(self, 
                          case1_features: torch.Tensor,
                          case2_features: torch.Tensor) -> torch.Tensor:
        """计算案例间的神经相似度"""
        emb1 = self.encode(case1_features)
        emb2 = self.encode(case2_features)
        
        # 余弦相似度
        return F.cosine_similarity(emb1, emb2, dim=0)


class CaseAdapter:
    """
    案例适配器
    
    将检索到的案例解决方案适配到新问题。
    """
    
    def __init__(self, adaptation_rules: List[Dict] = None):
        self.adaptation_rules = adaptation_rules or []
    
    def adapt(self,
             retrieved_case: MaterialCase,
             new_problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        适配案例到新问题
        
        Returns:
            适配后的解决方案
        """
        original_solution = retrieved_case.solution.copy()
        adapted_solution = original_solution.copy()
        
        # 基于差异进行适配
        problem_diff = self._compute_difference(
            retrieved_case.problem, new_problem
        )
        
        # 应用适配规则
        for rule in self.adaptation_rules:
            if self._rule_matches(rule, problem_diff):
                adapted_solution = self._apply_rule(rule, adapted_solution, problem_diff)
        
        # 数值参数插值
        adapted_solution = self._interpolate_parameters(
            retrieved_case.problem, new_problem, adapted_solution
        )
        
        return adapted_solution
    
    def _compute_difference(self, 
                           problem1: Dict,
                           problem2: Dict) -> Dict[str, Any]:
        """计算两个问题间的差异"""
        diff = {}
        all_keys = set(problem1.keys()) | set(problem2.keys())
        
        for key in all_keys:
            v1 = problem1.get(key)
            v2 = problem2.get(key)
            
            if v1 != v2:
                diff[key] = {'old': v1, 'new': v2}
        
        return diff
    
    def _rule_matches(self, rule: Dict, diff: Dict) -> bool:
        """检查规则是否匹配差异"""
        conditions = rule.get('conditions', [])
        for cond in conditions:
            if cond not in diff:
                return False
        return True
    
    def _apply_rule(self, 
                   rule: Dict, 
                   solution: Dict,
                   diff: Dict) -> Dict:
        """应用适配规则"""
        actions = rule.get('actions', [])
        
        for action in actions:
            target = action.get('target')
            operation = action.get('operation')
            value = action.get('value')
            
            if operation == 'scale' and target in solution:
                if isinstance(solution[target], (int, float)):
                    solution[target] *= value
            elif operation == 'add':
                solution[target] = solution.get(target, 0) + value
            elif operation == 'set':
                solution[target] = value
        
        return solution
    
    def _interpolate_parameters(self,
                               old_problem: Dict,
                               new_problem: Dict,
                               solution: Dict) -> Dict:
        """基于问题差异进行参数插值"""
        adapted = solution.copy()
        
        for key in solution:
            if isinstance(solution[key], (int, float)):
                # 检查是否有相关的目标参数
                for prob_key in new_problem:
                    if prob_key in old_problem:
                        old_val = old_problem[prob_key]
                        new_val = new_problem[prob_key]
                        
                        if isinstance(old_val, (int, float)) and \
                           isinstance(new_val, (int, float)) and old_val != 0:
                            # 线性插值
                            ratio = new_val / old_val
                            adapted[key] = solution[key] * ratio
        
        return adapted


class CaseBasedReasoner:
    """
    案例推理系统
    
    完整的CBR循环：检索-重用-修正-学习
    """
    
    def __init__(self,
                 similarity_func: Callable = SimilarityMetric.cosine,
                 use_neural_encoder: bool = False,
                 input_dim: int = None):
        self.retriever = CaseRetriever(similarity_func)
        self.adapter = CaseAdapter()
        self.use_neural_encoder = use_neural_encoder
        
        if use_neural_encoder and input_dim:
            self.neural_encoder = NeuralCaseEncoder(input_dim)
        else:
            self.neural_encoder = None
        
        self.case_library: List[MaterialCase] = []
        self.evaluation_history: List[Dict] = []
    
    def add_case(self, case: MaterialCase):
        """添加案例到库"""
        self.case_library.append(case)
        self.retriever.add_case(case)
    
    def solve(self,
             problem: Dict[str, Any],
             k: int = 3) -> Tuple[Dict[str, Any], List[Tuple[MaterialCase, float]], str]:
        """
        使用CBR解决新问题
        
        Returns:
            (解决方案, 检索到的案例列表, 推理说明)
        """
        # 1. 检索（Retrieve）
        query_case = MaterialCase(
            case_id="query",
            problem=problem
        )
        
        if self.neural_encoder:
            # 使用神经编码
            features = torch.tensor(query_case.to_vector(), dtype=torch.float32)
            with torch.no_grad():
                embedding = self.neural_encoder.encode(features)
            similar_cases = self.retriever.retrieve_by_features(
                embedding.numpy(), k
            )
        else:
            similar_cases = self.retriever.retrieve(query_case, k)
        
        if not similar_cases:
            return None, [], "No similar cases found."
        
        # 2. 重用（Reuse）
        best_case, best_similarity = similar_cases[0]
        adapted_solution = self.adapter.adapt(best_case, problem)
        
        # 3. 生成解释
        explanation = self._generate_explanation(
            problem, best_case, adapted_solution, similar_cases
        )
        
        return adapted_solution, similar_cases, explanation
    
    def revise(self,
              proposed_solution: Dict[str, Any],
              evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        修正（Revise）
        
        根据评估结果修正解决方案。
        """
        revised_solution = proposed_solution.copy()
        
        # 根据评估反馈调整
        if 'corrections' in evaluation:
            for correction in evaluation['corrections']:
                param = correction.get('parameter')
                new_value = correction.get('value')
                if param in revised_solution:
                    revised_solution[param] = new_value
        
        # 根据成功指标调整
        if 'success_metrics' in evaluation:
            metrics = evaluation['success_metrics']
            if 'accuracy' in metrics and metrics['accuracy'] < 0.8:
                # 解决方案需要重大调整
                revised_solution = self._major_revision(revised_solution, metrics)
        
        return revised_solution
    
    def _major_revision(self, 
                       solution: Dict[str, Any],
                       metrics: Dict[str, float]) -> Dict[str, Any]:
        """重大修订"""
        revised = solution.copy()
        
        # 基于指标进行系统性调整
        if 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            for key in revised:
                if isinstance(revised[key], (int, float)):
                    # 根据准确度调整参数
                    revised[key] *= (0.5 + accuracy)
        
        return revised
    
    def retain(self,
              problem: Dict[str, Any],
              solution: Dict[str, Any],
              outcome: Dict[str, Any],
              success_score: float):
        """
        保留（Retain）
        
        将新案例添加到案例库。
        """
        case_id = f"case_{len(self.case_library)}"
        
        new_case = MaterialCase(
            case_id=case_id,
            problem=problem,
            solution=solution,
            outcome=outcome,
            success_score=success_score,
            timestamp=len(self.case_library)
        )
        
        self.add_case(new_case)
        
        # 记录评估历史
        self.evaluation_history.append({
            'case_id': case_id,
            'success_score': success_score,
            'timestamp': new_case.timestamp
        })
        
        return case_id
    
    def evaluate_and_learn(self,
                          case_id: str,
                          actual_outcome: Dict[str, Any]) -> bool:
        """
        评估结果并学习
        
        根据实际结果更新案例库。
        """
        # 查找案例
        case = None
        for c in self.case_library:
            if c.case_id == case_id:
                case = c
                break
        
        if not case:
            return False
        
        # 计算成功分数
        success_score = self._compute_success_score(case.outcome, actual_outcome)
        
        # 更新案例
        case.outcome = actual_outcome
        case.success_score = success_score
        case.usage_count += 1
        
        # 如果成功分数很低，标记为待验证
        if success_score < 0.3:
            case.status = CaseStatus.PENDING
        
        return True
    
    def _compute_success_score(self,
                              expected: Dict[str, Any],
                              actual: Dict[str, Any]) -> float:
        """计算成功分数"""
        scores = []
        
        for key in expected:
            if key in actual:
                exp_val = expected[key]
                act_val = actual[key]
                
                if isinstance(exp_val, (int, float)) and \
                   isinstance(act_val, (int, float)):
                    # 数值比较：使用相对误差
                    if exp_val != 0:
                        error = abs(act_val - exp_val) / abs(exp_val)
                        score = max(0, 1 - error)
                    else:
                        score = 1.0 if act_val == 0 else 0.0
                    scores.append(score)
                elif exp_val == act_val:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_explanation(self,
                             problem: Dict[str, Any],
                             best_case: MaterialCase,
                             solution: Dict[str, Any],
                             similar_cases: List[Tuple[MaterialCase, float]]) -> str:
        """生成推理解释"""
        explanation = f"Case-Based Reasoning Solution:\n"
        explanation += f"  Problem: {problem}\n"
        explanation += f"  Most similar case: {best_case.case_id} "
        explanation += f"(similarity: {similar_cases[0][1]:.3f})\n"
        explanation += f"  Case problem: {best_case.problem}\n"
        explanation += f"  Case solution: {best_case.solution}\n"
        explanation += f"  Adapted solution: {solution}\n"
        
        if len(similar_cases) > 1:
            explanation += f"  Other similar cases:\n"
            for case, sim in similar_cases[1:]:
                explanation += f"    - {case.case_id} (similarity: {sim:.3f})\n"
        
        return explanation
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取案例库统计信息"""
        if not self.case_library:
            return {"message": "Empty case library"}
        
        success_scores = [c.success_score for c in self.case_library]
        usage_counts = [c.usage_count for c in self.case_library]
        
        return {
            "total_cases": len(self.case_library),
            "avg_success_score": np.mean(success_scores),
            "success_score_std": np.std(success_scores),
            "avg_usage_count": np.mean(usage_counts),
            "most_used_case": max(self.case_library, key=lambda c: c.usage_count).case_id,
            "best_case": max(self.case_library, key=lambda c: c.success_score).case_id,
            "status_distribution": {
                status.name: sum(1 for c in self.case_library if c.status == status)
                for status in CaseStatus
            }
        }


# ==================== 材料科学案例库 ====================

def create_sample_material_cases() -> List[MaterialCase]:
    """创建示例材料案例库"""
    
    cases = [
        MaterialCase(
            case_id="case_001",
            problem={
                "target_property": "high_thermal_conductivity",
                "temperature_range": "room_temp",
                "constraints": ["stable", "cost_effective"]
            },
            solution={
                "composition": {"Si": 0.5, "Ge": 0.5},
                "structure_type": "diamond",
                "synthesis_method": "czochralski"
            },
            outcome={
                "properties": {
                    "thermal_conductivity": 80,  # W/(m·K)
                    "band_gap": 0.8  # eV
                }
            },
            success_score=0.85,
            tags=["thermoelectric", "alloy"]
        ),
        
        MaterialCase(
            case_id="case_002",
            problem={
                "target_property": "large_band_gap",
                "band_gap_target": 3.0,
                "application": "optoelectronics"
            },
            solution={
                "composition": {"Ga": 0.5, "N": 0.5},
                "structure_type": "wurtzite",
                "synthesis_method": "mocvd"
            },
            outcome={
                "properties": {
                    "band_gap": 3.4,
                    "thermal_conductivity": 130
                }
            },
            success_score=0.92,
            tags=["semiconductor", "wide_bandgap"]
        ),
        
        MaterialCase(
            case_id="case_003",
            problem={
                "target_property": "high_electron_mobility",
                "carrier_type": "electron",
                "temperature": 300
            },
            solution={
                "composition": {"In": 0.53, "Ga": 0.47, "As": 1.0},
                "structure_type": "zincblende",
                "synthesis_method": "mbe"
            },
            outcome={
                "properties": {
                    "electron_mobility": 12000,  # cm²/(V·s)
                    "band_gap": 0.75
                }
            },
            success_score=0.88,
            tags=["high_mobility", "iii-v"]
        ),
        
        MaterialCase(
            case_id="case_004",
            problem={
                "target_property": "superconductivity",
                "critical_temp_target": 100,
                "pressure": "ambient"
            },
            solution={
                "composition": {"Y": 1, "Ba": 2, "Cu": 3, "O": 7},
                "structure_type": "perovskite",
                "synthesis_method": "solid_state"
            },
            outcome={
                "properties": {
                    "critical_temperature": 92,  # K
                    "critical_field": 120  # T
                }
            },
            success_score=0.75,
            tags=["superconductor", "cuprate"]
        ),
    ]
    
    return cases


if __name__ == "__main__":
    print("=" * 60)
    print("案例推理模块测试")
    print("=" * 60)
    
    # 创建案例推理系统
    cbr = CaseBasedReasoner(similarity_func=SimilarityMetric.cosine)
    
    # 添加示例案例
    cases = create_sample_material_cases()
    for case in cases:
        cbr.add_case(case)
    
    print(f"\n添加了 {len(cases)} 个案例到案例库")
    
    # 定义新问题
    new_problem = {
        "target_property": "high_thermal_conductivity",
        "temperature_range": "room_temp",
        "constraints": ["stable", "cost_effective"]
    }
    
    print(f"\n新问题: {new_problem}")
    
    # 使用CBR解决问题
    solution, similar_cases, explanation = cbr.solve(new_problem, k=2)
    
    print("\n检索到的相似案例:")
    for case, sim in similar_cases:
        print(f"  {case.case_id}: similarity={sim:.3f}")
    
    print(f"\n{explanation}")
    
    # 评估和学习
    actual_outcome = {
        "properties": {
            "thermal_conductivity": 85,
            "band_gap": 0.82
        }
    }
    
    # 保留新案例
    if solution:
        case_id = cbr.retain(new_problem, solution, actual_outcome, success_score=0.80)
        print(f"\n新案例已保留: {case_id}")
    
    # 查看统计信息
    stats = cbr.get_statistics()
    print(f"\n案例库统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
