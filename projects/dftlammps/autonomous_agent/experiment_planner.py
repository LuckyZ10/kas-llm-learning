"""
实验规划器模块
==============
实现自动化的实验规划功能，包括假设生成、实验设计优化、资源分配和风险评估。
"""

import asyncio
import itertools
import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.optimize import minimize


class ExperimentType(Enum):
    """实验类型枚举"""
    DFT_CALCULATION = "dft"
    MOLECULAR_DYNAMICS = "md"
    MONTE_CARLO = "mc"
    MACHINE_LEARNING = "ml"
    SYNTHESIS = "synthesis"
    CHARACTERIZATION = "characterization"


class HypothesisStatus(Enum):
    """假设状态枚举"""
    PROPOSED = auto()
    UNDER_TEST = auto()
    CONFIRMED = auto()
    REJECTED = auto()
    PENDING_REVISION = auto()


class ResourceType(Enum):
    """资源类型枚举"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    TIME = "time"
    COST = "cost"


@dataclass
class Hypothesis:
    """科学假设数据类"""
    id: str = field(default_factory=lambda: f"hyp_{random.randint(10000, 99999)}")
    statement: str = ""  # 假设陈述
    theoretical_basis: str = ""  # 理论基础
    expected_outcome: str = ""  # 预期结果
    testable_predictions: List[str] = field(default_factory=list)  # 可检验预测
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    priority: float = 0.5  # 优先级 0-1
    confidence: float = 0.5  # 置信度 0-1
    novelty_score: float = 0.5  # 新颖性分数
    experiments: List[str] = field(default_factory=list)  # 相关实验ID
    parent_hypothesis: Optional[str] = None  # 父假设ID
    created_at: datetime = field(default_factory=datetime.now)
    tested_at: Optional[datetime] = None
    result_summary: Optional[str] = None
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    
    def update_confidence(self, new_evidence: Dict[str, Any]):
        """根据新证据更新置信度"""
        if new_evidence.get("supports"):
            self.evidence_for.append(new_evidence.get("description", ""))
            # 贝叶斯更新
            prior = self.confidence
            likelihood = 0.8  # 假设证据可靠
            self.confidence = (likelihood * prior) / (
                likelihood * prior + (1 - likelihood) * (1 - prior)
            )
        else:
            self.evidence_against.append(new_evidence.get("description", ""))
            prior = self.confidence
            likelihood = 0.2
            self.confidence = (likelihood * prior) / (
                likelihood * prior + (1 - likelihood) * (1 - prior)
            )


@dataclass
class ExperimentalVariable:
    """实验变量"""
    name: str
    type: str  # "continuous", "discrete", "categorical"
    range: Union[Tuple[float, float], List[Any]]
    current_value: Any = None
    importance: float = 0.5
    
    def sample(self, method: str = "random") -> Any:
        """采样变量值"""
        if self.type == "continuous":
            low, high = self.range
            if method == "random":
                return np.random.uniform(low, high)
            elif method == "center":
                return (low + high) / 2
            elif method == "boundary":
                return random.choice([low, high])
        elif self.type == "discrete":
            if method == "random":
                return random.choice(self.range)
            elif method == "center":
                return self.range[len(self.range) // 2]
        elif self.type == "categorical":
            return random.choice(self.range)
        
        return None


@dataclass
class ExperimentDesign:
    """实验设计"""
    id: str = field(default_factory=lambda: f"exp_{random.randint(10000, 99999)}")
    name: str = ""
    description: str = ""
    experiment_type: ExperimentType = ExperimentType.DFT_CALCULATION
    variables: Dict[str, ExperimentalVariable] = field(default_factory=dict)
    hypothesis_id: Optional[str] = None
    design_matrix: List[Dict[str, Any]] = field(default_factory=list)  # 实验条件矩阵
    num_runs: int = 1
    replication: int = 1  # 重复次数
    controls: List[Dict[str, Any]] = field(default_factory=list)  # 对照组
    measurements: List[str] = field(default_factory=list)  # 需要测量的指标
    expected_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    estimated_cost: float = 0.0
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    optimization_metric: str = "accuracy"  # 优化目标


@dataclass
class ResourceAllocation:
    """资源分配"""
    experiment_id: str
    resources: Dict[ResourceType, float]
    start_time: datetime
    end_time: datetime
    priority: int = 5
    preemptible: bool = False  # 是否可抢占


@dataclass
class Risk:
    """风险项"""
    id: str = field(default_factory=lambda: f"risk_{random.randint(10000, 99999)}")
    description: str = ""
    category: str = ""  # "technical", "scientific", "operational", "financial"
    probability: float = 0.5  # 发生概率
    impact: float = 0.5  # 影响程度
    mitigation_strategy: str = ""
    contingency_plan: str = ""
    owner: str = ""  # 负责人
    status: str = "identified"  # "identified", "mitigated", "occurred", "resolved"


class HypothesisGenerator:
    """假设生成器"""
    
    def __init__(self):
        self.generation_strategies: List[Callable] = []
        self.knowledge_base: Dict[str, Any] = {}
        self.past_hypotheses: List[Hypothesis] = []
        
    def add_knowledge(self, domain: str, knowledge: Any):
        """添加领域知识"""
        self.knowledge_base[domain] = knowledge
        
    def register_strategy(self, strategy: Callable):
        """注册生成策略"""
        self.generation_strategies.append(strategy)
        
    def generate_hypotheses(self, research_question: str, num_hypotheses: int = 5) -> List[Hypothesis]:
        """
        生成科学假设
        
        基于研究问题和知识库生成可检验的假设。
        """
        hypotheses = []
        
        # 1. 基于知识库生成
        knowledge_based = self._generate_from_knowledge(research_question)
        hypotheses.extend(knowledge_based)
        
        # 2. 基于类比生成
        analogical = self._generate_by_analogy(research_question)
        hypotheses.extend(analogical)
        
        # 3. 基于模式识别生成
        pattern_based = self._generate_from_patterns(research_question)
        hypotheses.extend(pattern_based)
        
        # 4. 基于逆向思维生成
        reverse = self._generate_reverse(research_question)
        hypotheses.extend(reverse)
        
        # 评估和排序
        for h in hypotheses:
            h.novelty_score = self._calculate_novelty(h)
            h.priority = self._calculate_priority(h, research_question)
        
        # 选择最优的假设
        hypotheses.sort(key=lambda x: x.priority * x.novelty_score, reverse=True)
        
        selected = hypotheses[:num_hypotheses]
        self.past_hypotheses.extend(selected)
        
        return selected
    
    def _generate_from_knowledge(self, question: str) -> List[Hypothesis]:
        """基于知识库生成假设"""
        hypotheses = []
        
        # 分析研究问题
        keywords = self._extract_keywords(question)
        
        # 基于催化领域知识
        if any(kw in ["catalyst", "catalysis", "催化"] for kw in keywords):
            h1 = Hypothesis(
                statement="过渡金属掺杂可以提高催化剂的活性和选择性",
                theoretical_basis="d带中心理论预测，过渡金属掺杂可以调节d带中心位置",
                expected_outcome="过电位降低20%以上",
                testable_predictions=["计算不同掺杂浓度下的d带中心", "测量HER活性"]
            )
            hypotheses.append(h1)
            
            h2 = Hypothesis(
                statement="纳米结构可以增加活性位点密度",
                theoretical_basis="尺寸效应和表面原子配位不饱和增加",
                expected_outcome="质量活性提高3倍以上",
                testable_predictions=["合成不同尺寸的纳米颗粒", "测定活性位点数量"]
            )
            hypotheses.append(h2)
        
        # 基于电池材料知识
        if any(kw in ["battery", "lithium", "ion", "电池"] for kw in keywords):
            h1 = Hypothesis(
                statement="层状结构可以提供更快的离子传输通道",
                theoretical_basis="二维扩散路径缩短离子传输距离",
                expected_outcome="倍率性能提高50%",
                testable_predictions=["计算离子扩散系数", "测量不同倍率下的容量"]
            )
            hypotheses.append(h1)
        
        return hypotheses
    
    def _generate_by_analogy(self, question: str) -> List[Hypothesis]:
        """基于类比生成假设"""
        hypotheses = []
        
        # 从已知材料推断新材料
        h = Hypothesis(
            statement="类似钙钛矿结构的材料可能具有优异的光催化性能",
            theoretical_basis="钙钛矿结构已知的优异光电性质可以迁移",
            expected_outcome="可见光响应范围扩展到600nm",
            testable_predictions=["计算能带结构", "测试光催化活性"],
            novelty_score=0.7
        )
        hypotheses.append(h)
        
        return hypotheses
    
    def _generate_from_patterns(self, question: str) -> List[Hypothesis]:
        """基于模式识别生成假设"""
        hypotheses = []
        
        # 分析历史成功模式
        h = Hypothesis(
            statement="具有特定电子构型的元素组合会产生协同效应",
            theoretical_basis="电子态杂化可以优化反应中间体吸附能",
            expected_outcome="协同因子大于1.5",
            testable_predictions=["构建不同元素组合", "计算吸附能线性关系"]
        )
        hypotheses.append(h)
        
        return hypotheses
    
    def _generate_reverse(self, question: str) -> List[Hypothesis]:
        """基于逆向思维生成假设"""
        hypotheses = []
        
        # 反向思考
        h = Hypothesis(
            statement="故意引入缺陷可能是提高性能的关键",
            theoretical_basis="缺陷可以创造新的活性位点并调节电子结构",
            expected_outcome="缺陷样品活性提高2倍",
            testable_predictions=["制备不同缺陷浓度的样品", "表征缺陷类型和浓度"]
        )
        hypotheses.append(h)
        
        return hypotheses
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化的关键词提取
        keywords = []
        important_terms = [
            "catalyst", "catalysis", "battery", "solar", "energy", "material",
            "纳米", "催化", "电池", "能源", "材料", "electronic", "structure"
        ]
        text_lower = text.lower()
        for term in important_terms:
            if term in text_lower:
                keywords.append(term)
        return keywords
    
    def _calculate_novelty(self, hypothesis: Hypothesis) -> float:
        """计算假设新颖性"""
        if not self.past_hypotheses:
            return 1.0
        
        # 与历史假设比较
        similarities = []
        for past in self.past_hypotheses:
            sim = self._text_similarity(hypothesis.statement, past.statement)
            similarities.append(sim)
        
        # 新颖性 = 1 - 最大相似度
        return 1.0 - max(similarities) if similarities else 1.0
    
    def _calculate_priority(self, hypothesis: Hypothesis, question: str) -> float:
        """计算假设优先级"""
        # 基于与问题的相关性和潜在影响
        relevance = self._text_similarity(hypothesis.statement, question)
        impact = hypothesis.confidence * 0.5 + hypothesis.novelty_score * 0.5
        return (relevance + impact) / 2
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简化的Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0


class DesignOptimizer:
    """实验设计优化器"""
    
    def __init__(self):
        self.optimization_methods: Dict[str, Callable] = {}
        self.design_history: List[ExperimentDesign] = []
        
    def register_method(self, name: str, method: Callable):
        """注册优化方法"""
        self.optimization_methods[name] = method
        
    def optimize_design(self, 
                       hypothesis: Hypothesis,
                       variables: List[ExperimentalVariable],
                       constraints: Dict[str, Any],
                       objective: str = "information_gain") -> ExperimentDesign:
        """
        优化实验设计
        
        使用贝叶斯优化、遗传算法等方法优化实验条件。
        """
        design = ExperimentDesign(
            name=f"测试_{hypothesis.id}",
            description=f"验证假设: {hypothesis.statement[:50]}...",
            hypothesis_id=hypothesis.id,
            variables={v.name: v for v in variables}
        )
        
        # 选择优化策略
        if objective == "information_gain":
            design_matrix = self._optimize_information_gain(variables, constraints)
        elif objective == "cost_efficiency":
            design_matrix = self._optimize_cost_efficiency(variables, constraints)
        elif objective == "coverage":
            design_matrix = self._optimize_coverage(variables, constraints)
        else:
            design_matrix = self._latin_hypercube_sampling(variables, constraints.get("num_samples", 10))
        
        design.design_matrix = design_matrix
        design.num_runs = len(design_matrix)
        
        # 计算资源需求
        design.resource_requirements = self._estimate_resources(design)
        design.estimated_cost = self._estimate_cost(design)
        
        self.design_history.append(design)
        return design
    
    def _optimize_information_gain(self, variables: List[ExperimentalVariable], 
                                    constraints: Dict) -> List[Dict]:
        """基于信息增益优化"""
        num_samples = constraints.get("num_samples", 10)
        
        # 使用拉丁超立方采样确保空间覆盖
        samples = []
        continuous_vars = [v for v in variables if v.type == "continuous"]
        
        if continuous_vars:
            # 拉丁超立方采样
            lhs_samples = self._latin_hypercube_sampling(continuous_vars, num_samples)
            
            # 添加离散和类别变量
            for sample in lhs_samples:
                for v in variables:
                    if v.type != "continuous":
                        sample[v.name] = v.sample()
                samples.append(sample)
        else:
            # 全因子设计简化版
            samples = self._factorial_design(variables, constraints)
        
        return samples
    
    def _optimize_cost_efficiency(self, variables: List[ExperimentalVariable],
                                   constraints: Dict) -> List[Dict]:
        """基于成本效率优化"""
        max_budget = constraints.get("max_budget", 1000)
        cost_per_run = constraints.get("cost_per_run", 10)
        
        max_runs = int(max_budget / cost_per_run)
        
        # 使用较少的样本点但优化其位置
        num_samples = min(max_runs, 20)
        
        return self._latin_hypercube_sampling(variables, num_samples)
    
    def _optimize_coverage(self, variables: List[ExperimentalVariable],
                           constraints: Dict) -> List[Dict]:
        """优化空间覆盖"""
        # 使用网格采样和随机采样结合
        grid_samples = self._grid_sampling(variables, constraints.get("grid_size", 3))
        random_samples = self._random_sampling(variables, constraints.get("num_random", 5))
        
        return grid_samples + random_samples
    
    def _latin_hypercube_sampling(self, variables: List[ExperimentalVariable], 
                                   num_samples: int) -> List[Dict]:
        """拉丁超立方采样"""
        samples = []
        n_vars = len(variables)
        
        # 生成LHS矩阵
        permutations = [np.random.permutation(num_samples) for _ in range(n_vars)]
        
        for i in range(num_samples):
            sample = {}
            for j, var in enumerate(variables):
                if var.type == "continuous":
                    low, high = var.range
                    # LHS采样点
                    point = (permutations[j][i] + np.random.random()) / num_samples
                    sample[var.name] = low + point * (high - low)
                else:
                    sample[var.name] = var.sample()
            samples.append(sample)
        
        return samples
    
    def _factorial_design(self, variables: List[ExperimentalVariable], 
                          constraints: Dict) -> List[Dict]:
        """全因子设计"""
        levels = constraints.get("levels", 2)
        
        # 为每个变量生成水平
        var_levels = []
        for v in variables:
            if v.type == "continuous":
                low, high = v.range
                var_levels.append([(low + high) / 2 if levels == 1 else 
                                  low + i * (high - low) / (levels - 1) 
                                  for i in range(levels)])
            elif v.type == "discrete":
                var_levels.append(v.range[:levels])
            else:
                var_levels.append(v.range[:levels])
        
        # 生成全因子组合
        combinations = list(itertools.product(*var_levels))
        
        samples = []
        for combo in combinations:
            sample = {v.name: combo[i] for i, v in enumerate(variables)}
            samples.append(sample)
        
        return samples
    
    def _grid_sampling(self, variables: List[ExperimentalVariable], 
                       grid_size: int) -> List[Dict]:
        """网格采样"""
        return self._factorial_design(variables, {"levels": grid_size})
    
    def _random_sampling(self, variables: List[ExperimentalVariable], 
                         num_samples: int) -> List[Dict]:
        """随机采样"""
        samples = []
        for _ in range(num_samples):
            sample = {v.name: v.sample() for v in variables}
            samples.append(sample)
        return samples
    
    def _estimate_resources(self, design: ExperimentDesign) -> Dict[ResourceType, float]:
        """估计资源需求"""
        resources = {
            ResourceType.CPU: design.num_runs * 4,  # 4核小时/次
            ResourceType.MEMORY: design.num_runs * 8,  # 8GB/次
            ResourceType.TIME: design.num_runs * 2,  # 2小时/次
        }
        
        if design.experiment_type == ExperimentType.MOLECULAR_DYNAMICS:
            resources[ResourceType.GPU] = design.num_runs * 1
        
        return resources
    
    def _estimate_cost(self, design: ExperimentDesign) -> float:
        """估计成本"""
        base_cost = design.num_runs * 10  # 基础成本
        
        # 根据类型调整
        type_multipliers = {
            ExperimentType.DFT_CALCULATION: 2.0,
            ExperimentType.MOLECULAR_DYNAMICS: 3.0,
            ExperimentType.MONTE_CARLO: 1.5,
            ExperimentType.MACHINE_LEARNING: 1.0,
        }
        
        return base_cost * type_multipliers.get(design.experiment_type, 1.0)


class ResourceScheduler:
    """资源调度器"""
    
    def __init__(self):
        self.available_resources: Dict[ResourceType, float] = {
            ResourceType.CPU: 100,
            ResourceType.GPU: 10,
            ResourceType.MEMORY: 1000,
            ResourceType.STORAGE: 10000
        }
        self.allocated_resources: Dict[str, ResourceAllocation] = {}
        self.schedule: List[ResourceAllocation] = []
        
    def set_resources(self, resources: Dict[ResourceType, float]):
        """设置可用资源"""
        self.available_resources.update(resources)
        
    def schedule_experiments(self, designs: List[ExperimentDesign],
                            constraints: Optional[Dict] = None) -> List[ResourceAllocation]:
        """
        调度实验执行
        
        优化实验执行顺序和资源分配。
        """
        allocations = []
        current_time = datetime.now()
        
        # 按优先级和依赖关系排序
        sorted_designs = self._topological_sort(designs)
        
        for design in sorted_designs:
            # 检查资源可用性
            if self._check_resources_available(design.resource_requirements):
                # 分配资源
                allocation = ResourceAllocation(
                    experiment_id=design.id,
                    resources=design.resource_requirements,
                    start_time=current_time,
                    end_time=current_time + design.expected_duration,
                    priority=5
                )
                allocations.append(allocation)
                self.allocated_resources[design.id] = allocation
                
                # 更新可用资源
                self._allocate_resources(design.resource_requirements)
                
                # 更新时间
                current_time = allocation.end_time
            else:
                # 资源不足，延迟调度
                delay = self._estimate_wait_time(design.resource_requirements)
                current_time += delay
                
                allocation = ResourceAllocation(
                    experiment_id=design.id,
                    resources=design.resource_requirements,
                    start_time=current_time,
                    end_time=current_time + design.expected_duration,
                    priority=5
                )
                allocations.append(allocation)
                current_time = allocation.end_time
        
        self.schedule.extend(allocations)
        return allocations
    
    def _topological_sort(self, designs: List[ExperimentDesign]) -> List[ExperimentDesign]:
        """拓扑排序处理依赖"""
        # 简化的拓扑排序
        design_map = {d.id: d for d in designs}
        sorted_designs = []
        visited = set()
        
        def visit(design):
            if design.id in visited:
                return
            visited.add(design.id)
            
            for dep_id in design.dependencies:
                if dep_id in design_map:
                    visit(design_map[dep_id])
            
            sorted_designs.append(design)
        
        for design in designs:
            visit(design)
        
        return sorted_designs
    
    def _check_resources_available(self, requirements: Dict[ResourceType, float]) -> bool:
        """检查资源是否可用"""
        for resource_type, amount in requirements.items():
            if resource_type in self.available_resources:
                if self.available_resources[resource_type] < amount:
                    return False
        return True
    
    def _allocate_resources(self, requirements: Dict[ResourceType, float]):
        """分配资源"""
        for resource_type, amount in requirements.items():
            if resource_type in self.available_resources:
                self.available_resources[resource_type] -= amount
    
    def _estimate_wait_time(self, requirements: Dict[ResourceType, float]) -> timedelta:
        """估计等待时间"""
        # 简化的估计
        return timedelta(hours=1)
    
    def release_resources(self, experiment_id: str):
        """释放资源"""
        if experiment_id in self.allocated_resources:
            allocation = self.allocated_resources[experiment_id]
            for resource_type, amount in allocation.resources.items():
                if resource_type in self.available_resources:
                    self.available_resources[resource_type] += amount
            del self.allocated_resources[experiment_id]
    
    def get_utilization(self) -> Dict[str, float]:
        """获取资源利用率"""
        utilization = {}
        for resource_type, total in self.available_resources.items():
            allocated = sum(
                a.resources.get(resource_type, 0) 
                for a in self.allocated_resources.values()
            )
            utilization[resource_type.value] = allocated / (allocated + total) if total > 0 else 0
        return utilization


class RiskAssessor:
    """风险评估器"""
    
    def __init__(self):
        self.risk_categories: Dict[str, List[str]] = {
            "technical": ["计算失败", "收敛问题", "精度不足"],
            "scientific": ["假设不成立", "效应太小", "干扰因素"],
            "operational": ["资源不足", "时间延迟", "数据丢失"],
            "financial": ["预算超支", "设备损坏", "重复实验"]
        }
        self.known_risks: List[Risk] = []
        
    def assess_experiment(self, design: ExperimentDesign) -> List[Risk]:
        """
        评估实验风险
        
        识别潜在风险并制定缓解策略。
        """
        risks = []
        
        # 技术风险
        if design.experiment_type == ExperimentType.DFT_CALCULATION:
            risks.append(Risk(
                description="DFT计算可能不收敛",
                category="technical",
                probability=0.3,
                impact=0.6,
                mitigation_strategy="使用更好的初始猜测，增加迭代次数",
                contingency_plan="切换到更稳定的计算方法"
            ))
        
        if design.experiment_type == ExperimentType.MOLECULAR_DYNAMICS:
            risks.append(Risk(
                description="MD模拟可能遇到数值不稳定性",
                category="technical",
                probability=0.2,
                impact=0.5,
                mitigation_strategy="使用适当的时间步长，添加约束条件",
                contingency_plan="减小时间步长重新运行"
            ))
        
        # 科学风险
        if design.hypothesis_id:
            risks.append(Risk(
                description="假设可能不成立",
                category="scientific",
                probability=0.5,
                impact=0.7,
                mitigation_strategy="设计对照实验，准备替代假设",
                contingency_plan="回到假设生成阶段重新分析"
            ))
        
        # 操作风险
        if design.estimated_cost > 1000:
            risks.append(Risk(
                description="实验成本可能超预算",
                category="financial",
                probability=0.4,
                impact=0.5,
                mitigation_strategy="分阶段执行，先进行小规模测试",
                contingency_plan="申请额外预算或简化实验设计"
            ))
        
        # 资源风险
        if design.expected_duration > timedelta(hours=48):
            risks.append(Risk(
                description="长时间运行可能遇到系统故障",
                category="operational",
                probability=0.2,
                impact=0.8,
                mitigation_strategy="设置检查点，启用自动备份",
                contingency_plan="从最近检查点恢复"
            ))
        
        # 计算整体风险评分
        for risk in risks:
            risk.probability = self._refine_probability(risk, design)
            risk.impact = self._refine_impact(risk, design)
        
        self.known_risks.extend(risks)
        return risks
    
    def _refine_probability(self, risk: Risk, design: ExperimentDesign) -> float:
        """精细化风险概率"""
        base_prob = risk.probability
        
        # 根据实验设计调整
        if risk.category == "technical" and design.num_runs > 10:
            base_prob *= 1.2  # 更多运行意味着更多失败机会
        
        if risk.category == "operational" and design.dependencies:
            base_prob *= 1.1  # 依赖增加复杂性
        
        return min(base_prob, 1.0)
    
    def _refine_impact(self, risk: Risk, design: ExperimentDesign) -> float:
        """精细化风险影响"""
        base_impact = risk.impact
        
        # 根据资源投入调整
        if design.estimated_cost > 1000:
            base_impact *= 1.1
        
        return min(base_impact, 1.0)
    
    def calculate_risk_score(self, risks: List[Risk]) -> float:
        """计算总体风险分数"""
        if not risks:
            return 0.0
        
        total_score = sum(r.probability * r.impact for r in risks)
        return total_score / len(risks)
    
    def get_mitigation_plan(self, risks: List[Risk]) -> Dict[str, List[str]]:
        """获取风险缓解计划"""
        plan = {}
        
        for risk in risks:
            if risk.category not in plan:
                plan[risk.category] = []
            plan[risk.category].append({
                "risk": risk.description,
                "mitigation": risk.mitigation_strategy,
                "contingency": risk.contingency_plan
            })
        
        return plan


class ExperimentPlanner:
    """
    实验规划器主类
    
    整合假设生成、设计优化、资源调度和风险评估功能。
    """
    
    def __init__(self):
        self.hypothesis_generator = HypothesisGenerator()
        self.design_optimizer = DesignOptimizer()
        self.resource_scheduler = ResourceScheduler()
        self.risk_assessor = RiskAssessor()
        
        self.plans: Dict[str, Dict] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def create_experiment_plan(self,
                                     research_question: str,
                                     available_resources: Optional[Dict] = None,
                                     constraints: Optional[Dict] = None) -> Dict:
        """
        创建完整的实验计划
        
        从研究问题到可执行计划的完整流程。
        """
        self.logger.info(f"开始为研究问题创建实验计划: {research_question[:50]}...")
        
        plan_id = f"plan_{random.randint(100000, 999999)}"
        
        # 1. 生成假设
        self.logger.info("生成科学假设...")
        hypotheses = self.hypothesis_generator.generate_hypotheses(
            research_question, 
            num_hypotheses=constraints.get("num_hypotheses", 5) if constraints else 5
        )
        self.logger.info(f"生成 {len(hypotheses)} 个假设")
        
        # 2. 为每个假设设计实验
        self.logger.info("优化实验设计...")
        designs = []
        for hypothesis in hypotheses[:3]:  # 测试前3个假设
            variables = self._define_variables_for_hypothesis(hypothesis)
            design = self.design_optimizer.optimize_design(
                hypothesis, 
                variables,
                constraints or {},
                objective=constraints.get("objective", "information_gain") if constraints else "information_gain"
            )
            designs.append(design)
        
        # 3. 评估风险
        self.logger.info("评估实验风险...")
        all_risks = []
        for design in designs:
            risks = self.risk_assessor.assess_experiment(design)
            all_risks.extend(risks)
        
        # 4. 调度资源
        if available_resources:
            self.resource_scheduler.set_resources(available_resources)
        
        self.logger.info("调度实验执行...")
        allocations = self.resource_scheduler.schedule_experiments(designs)
        
        # 5. 组装计划
        plan = {
            "id": plan_id,
            "research_question": research_question,
            "created_at": datetime.now(),
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "confidence": h.confidence,
                    "priority": h.priority,
                    "novelty": h.novelty_score
                }
                for h in hypotheses
            ],
            "experiments": [
                {
                    "id": d.id,
                    "name": d.name,
                    "type": d.experiment_type.value,
                    "num_runs": d.num_runs,
                    "estimated_cost": d.estimated_cost,
                    "variables": list(d.variables.keys()),
                    "hypothesis_id": d.hypothesis_id
                }
                for d in designs
            ],
            "schedule": [
                {
                    "experiment_id": a.experiment_id,
                    "start_time": a.start_time.isoformat(),
                    "end_time": a.end_time.isoformat(),
                    "resources": {k.value: v for k, v in a.resources.items()}
                }
                for a in allocations
            ],
            "risks": [
                {
                    "description": r.description,
                    "category": r.category,
                    "probability": r.probability,
                    "impact": r.impact,
                    "score": r.probability * r.impact
                }
                for r in all_risks
            ],
            "total_estimated_cost": sum(d.estimated_cost for d in designs),
            "total_duration": sum((a.end_time - a.start_time).total_seconds() 
                                  for a in allocations) / 3600,  # 小时
            "overall_risk_score": self.risk_assessor.calculate_risk_score(all_risks),
            "mitigation_plan": self.risk_assessor.get_mitigation_plan(all_risks)
        }
        
        self.plans[plan_id] = plan
        self.logger.info(f"实验计划 {plan_id} 创建完成")
        
        return plan
    
    def _define_variables_for_hypothesis(self, hypothesis: Hypothesis) -> List[ExperimentalVariable]:
        """为假设定义实验变量"""
        variables = []
        
        # 基于假设内容推断变量
        statement = hypothesis.statement.lower()
        
        if "doping" in statement or "掺杂" in statement:
            variables.append(ExperimentalVariable(
                name="doping_concentration",
                type="continuous",
                range=(0.0, 0.3),
                importance=0.9
            ))
            variables.append(ExperimentalVariable(
                name="dopant_element",
                type="categorical",
                range=["Fe", "Co", "Ni", "Cu", "Zn"],
                importance=0.8
            ))
        
        if "temperature" in statement or "温度" in statement:
            variables.append(ExperimentalVariable(
                name="temperature",
                type="continuous",
                range=(273, 1273),
                importance=0.7
            ))
        
        if "pressure" in statement or "压力" in statement:
            variables.append(ExperimentalVariable(
                name="pressure",
                type="continuous",
                range=(0.1, 100),
                importance=0.6
            ))
        
        if not variables:
            # 默认变量
            variables.append(ExperimentalVariable(
                name="composition",
                type="categorical",
                range=["A", "B", "C"],
                importance=0.5
            ))
        
        return variables
    
    def get_plan_summary(self, plan_id: str) -> Dict:
        """获取计划摘要"""
        if plan_id not in self.plans:
            return {"error": "Plan not found"}
        
        plan = self.plans[plan_id]
        
        return {
            "id": plan["id"],
            "hypothesis_count": len(plan["hypotheses"]),
            "experiment_count": len(plan["experiments"]),
            "total_cost": plan["total_estimated_cost"],
            "total_duration_hours": plan["total_duration"],
            "risk_score": plan["overall_risk_score"],
            "top_hypotheses": sorted(
                plan["hypotheses"], 
                key=lambda x: x["priority"] * x["confidence"], 
                reverse=True
            )[:3]
        }
    
    def update_plan(self, plan_id: str, updates: Dict) -> Dict:
        """更新计划"""
        if plan_id not in self.plans:
            return {"error": "Plan not found"}
        
        self.plans[plan_id].update(updates)
        return self.plans[plan_id]


if __name__ == "__main__":
    # 测试代码
    async def test_planner():
        planner = ExperimentPlanner()
        
        # 创建一个实验计划
        plan = await planner.create_experiment_plan(
            research_question="发现用于水分解的高效过渡金属催化剂",
            available_resources={
                ResourceType.CPU: 200,
                ResourceType.GPU: 20,
                ResourceType.MEMORY: 2000
            },
            constraints={
                "num_hypotheses": 5,
                "max_budget": 5000,
                "objective": "information_gain"
            }
        )
        
        print("\n实验计划创建成功!")
        print(f"计划ID: {plan['id']}")
        print(f"假设数量: {len(plan['hypotheses'])}")
        print(f"实验数量: {len(plan['experiments'])}")
        print(f"预计总成本: ${plan['total_estimated_cost']:.2f}")
        print(f"预计总时长: {plan['total_duration']:.1f} 小时")
        print(f"整体风险分数: {plan['overall_risk_score']:.2f}")
        
        print("\n假设列表:")
        for i, h in enumerate(plan['hypotheses'][:3], 1):
            print(f"  {i}. {h['statement'][:60]}...")
            print(f"     置信度: {h['confidence']:.2f}, 优先级: {h['priority']:.2f}")
    
    asyncio.run(test_planner())
