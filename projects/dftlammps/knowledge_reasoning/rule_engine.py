"""
规则引擎模块 - Rule Engine for Materials Science

实现基于规则的专家系统，支持前向/后向链式推理、
冲突消解和不确定推理。
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CertaintyFactor:
    """确定性因子 - 用于不确定推理"""
    
    def __init__(self, value: float):
        self.value = max(-1.0, min(1.0, value))
    
    def __add__(self, other: 'CertaintyFactor') -> 'CertaintyFactor':
        """组合两个独立证据的确定性因子"""
        x, y = self.value, other.value
        if x > 0 and y > 0:
            return CertaintyFactor(x + y - x * y)
        elif x < 0 and y < 0:
            return CertaintyFactor(x + y + x * y)
        else:
            return CertaintyFactor((x + y) / (1 - min(abs(x), abs(y))))
    
    def __mul__(self, other: Union['CertaintyFactor', float]) -> 'CertaintyFactor':
        """传递不确定性（规则应用）"""
        if isinstance(other, CertaintyFactor):
            return CertaintyFactor(self.value * other.value)
        return CertaintyFactor(self.value * other)
    
    def __repr__(self):
        return f"CF({self.value:.3f})"


class Fact:
    """事实/断言"""
    
    def __init__(self,
                 predicate: str,
                 args: List[Any],
                 cf: CertaintyFactor = None,
                 source: Optional[str] = None,
                 timestamp: int = 0):
        self.predicate = predicate
        self.args = args
        self.cf = cf or CertaintyFactor(1.0)
        self.source = source
        self.timestamp = timestamp
    
    def __hash__(self):
        return hash((self.predicate, tuple(self.args)))
    
    def __eq__(self, other):
        if not isinstance(other, Fact):
            return False
        return self.predicate == other.predicate and self.args == other.args
    
    def __repr__(self):
        args_str = ', '.join(str(arg) for arg in self.args)
        return f"{self.predicate}({args_str}) [{self.cf}]"
    
    def matches(self, pattern: 'Fact') -> bool:
        """检查是否与模式匹配（支持变量）"""
        if self.predicate != pattern.predicate:
            return False
        if len(self.args) != len(pattern.args):
            return False
        
        for self_arg, pat_arg in zip(self.args, pattern.args):
            # 变量可以匹配任何值
            if isinstance(pat_arg, str) and pat_arg.startswith('?'):
                continue
            if self_arg != pat_arg:
                return False
        
        return True


class Condition:
    """规则条件"""
    
    def __init__(self,
                 predicate: str,
                 args: List[Any],
                 negated: bool = False):
        self.predicate = predicate
        self.args = args
        self.negated = negated
    
    def __repr__(self):
        args_str = ', '.join(str(arg) for arg in self.args)
        prefix = "not " if self.negated else ""
        return f"{prefix}{self.predicate}({args_str})"


class Rule:
    """
    产生式规则
    
    IF condition1 AND condition2 ... THEN conclusion
    """
    
    def __init__(self,
                 name: str,
                 conditions: List[Condition],
                 conclusion: Fact,
                 cf: CertaintyFactor = None,
                 priority: int = 0,
                 meta: Dict[str, Any] = None):
        self.name = name
        self.conditions = conditions
        self.conclusion = conclusion
        self.cf = cf or CertaintyFactor(1.0)
        self.priority = priority
        self.meta = meta or {}
        self.usage_count = 0
    
    def __repr__(self):
        conds_str = ' AND '.join(str(c) for c in self.conditions)
        return f"Rule {self.name}: IF {conds_str} THEN {self.conclusion}"
    
    def is_triggered(self, facts: Set[Fact]) -> Tuple[bool, CertaintyFactor, Dict[str, Any]]:
        """
        检查规则是否被触发
        
        Returns:
            (是否触发, 组合确定性因子, 变量绑定)
        """
        bindings = {}
        combined_cf = CertaintyFactor(1.0)
        
        for condition in self.conditions:
            matched, cf, new_bindings = self._match_condition(
                condition, facts, bindings
            )
            
            if not matched:
                return False, CertaintyFactor(0.0), {}
            
            # 更新绑定
            bindings.update(new_bindings)
            
            # 组合确定性因子（使用AND）
            if condition.negated:
                combined_cf = combined_cf * CertaintyFactor(1 - cf.value)
            else:
                combined_cf = combined_cf * cf
        
        return True, combined_cf, bindings
    
    def _match_condition(self,
                        condition: Condition,
                        facts: Set[Fact],
                        existing_bindings: Dict[str, Any]) -> Tuple[bool, CertaintyFactor, Dict[str, Any]]:
        """匹配单个条件"""
        best_cf = CertaintyFactor(0.0)
        best_bindings = {}
        
        for fact in facts:
            if fact.predicate != condition.predicate:
                continue
            
            # 尝试匹配参数
            match, bindings = self._match_args(
                condition.args, fact.args, existing_bindings
            )
            
            if match:
                if fact.cf.value > best_cf.value:
                    best_cf = fact.cf
                    best_bindings = bindings
        
        if best_cf.value > 0:
            return True, best_cf, best_bindings
        
        return False, CertaintyFactor(0.0), {}
    
    def _match_args(self,
                   pattern_args: List[Any],
                   fact_args: List[Any],
                   bindings: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """匹配参数并收集变量绑定"""
        new_bindings = bindings.copy()
        
        for pat_arg, fact_arg in zip(pattern_args, fact_args):
            if isinstance(pat_arg, str) and pat_arg.startswith('?'):
                # 变量
                if pat_arg in new_bindings:
                    if new_bindings[pat_arg] != fact_arg:
                        return False, {}
                else:
                    new_bindings[pat_arg] = fact_arg
            elif pat_arg != fact_arg:
                return False, {}
        
        return True, new_bindings
    
    def apply(self, bindings: Dict[str, Any]) -> Fact:
        """应用绑定生成结论"""
        # 替换结论中的变量
        new_args = []
        for arg in self.conclusion.args:
            if isinstance(arg, str) and arg.startswith('?'):
                new_args.append(bindings.get(arg, arg))
            else:
                new_args.append(arg)
        
        return Fact(
            self.conclusion.predicate,
            new_args,
            self.conclusion.cf
        )


class ConflictResolutionStrategy(Enum):
    """冲突消解策略"""
    PRIORITY = auto()           # 按优先级
    RECENCY = auto()            # 按最近使用
    SPECIFICITY = auto()        # 按特殊性（条件数量）
    CERTAINTY = auto()          # 按确定性因子


class RuleEngine:
    """
    规则引擎
    
    实现前向链式推理的专家系统。
    """
    
    def __init__(self,
                 conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.PRIORITY):
        self.rules: List[Rule] = []
        self.facts: Set[Fact] = set()
        self.conflict_resolution = conflict_resolution
        self.inference_log: List[Dict[str, Any]] = []
        self.timestamp = 0
        self.max_iterations = 1000
    
    def add_rule(self, rule: Rule):
        """添加规则"""
        self.rules.append(rule)
    
    def add_fact(self, fact: Fact):
        """添加事实"""
        fact.timestamp = self.timestamp
        self.timestamp += 1
        
        # 检查是否已有相同事实（考虑不确定性）
        existing = self._find_existing_fact(fact)
        if existing:
            # 合并确定性因子
            combined_cf = existing.cf + fact.cf
            existing.cf = combined_cf
        else:
            self.facts.add(fact)
    
    def _find_existing_fact(self, fact: Fact) -> Optional[Fact]:
        """查找已存在的事实"""
        for f in self.facts:
            if f.predicate == fact.predicate and f.args == fact.args:
                return f
        return None
    
    def query(self, pattern: Fact, min_cf: float = 0.0) -> List[Fact]:
        """
        查询符合模式的事实
        
        Args:
            pattern: 查询模式（可包含变量）
            min_cf: 最小确定性因子阈值
        """
        results = []
        for fact in self.facts:
            if fact.cf.value >= min_cf and fact.matches(pattern):
                results.append(fact)
        return results
    
    def forward_chain(self, goal: Optional[Fact] = None) -> bool:
        """
        前向链式推理
        
        从已知事实出发，应用规则推导新事实，
        直到达到目标或没有新事实可推导。
        
        Returns:
            是否达到目标（如果有目标）
        """
        for iteration in range(self.max_iterations):
            # 找到所有可触发的规则
            triggered_rules = []
            
            for rule in self.rules:
                triggered, cf, bindings = rule.is_triggered(self.facts)
                if triggered:
                    # 检查结论是否已知
                    conclusion = rule.apply(bindings)
                    conclusion.cf = conclusion.cf * cf * rule.cf
                    
                    if conclusion.cf.value > 0.1:  # 最小确定性阈值
                        existing = self._find_existing_fact(conclusion)
                        if not existing or existing.cf.value < conclusion.cf.value:
                            triggered_rules.append((rule, conclusion, bindings))
            
            if not triggered_rules:
                break  # 没有可触发的规则
            
            # 冲突消解
            selected_rule, conclusion, bindings = self._resolve_conflicts(
                triggered_rules
            )
            
            # 应用选中的规则
            self.add_fact(conclusion)
            selected_rule.usage_count += 1
            
            # 记录推理
            self.inference_log.append({
                'iteration': iteration,
                'rule': selected_rule.name,
                'conclusion': conclusion,
                'bindings': bindings
            })
            
            # 检查是否达到目标
            if goal and conclusion.matches(goal):
                return True
        
        return goal is None
    
    def backward_chain(self, goal: Fact, visited: Set[str] = None) -> Tuple[bool, CertaintyFactor, List[Dict]]:
        """
        后向链式推理
        
        从目标出发，反向寻找支持该目标的证据。
        
        Returns:
            (是否成功, 确定性因子, 证明链)
        """
        if visited is None:
            visited = set()
        
        # 检查目标是否已在事实中
        for fact in self.facts:
            if fact.matches(goal):
                return True, fact.cf, [{'type': 'fact', 'fact': fact}]
        
        # 避免循环
        goal_key = f"{goal.predicate}:{goal.args}"
        if goal_key in visited:
            return False, CertaintyFactor(0.0), []
        visited.add(goal_key)
        
        # 查找可以推导出目标的规则
        applicable_rules = []
        for rule in self.rules:
            if rule.conclusion.matches(goal):
                applicable_rules.append(rule)
        
        # 按优先级排序
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        # 尝试每个规则
        for rule in applicable_rules:
            # 证明规则的所有条件
            condition_cfs = []
            proof_chain = [{'type': 'rule', 'rule': rule}]
            all_proved = True
            
            for condition in rule.conditions:
                # 创建条件目标
                cond_goal = Fact(condition.predicate, condition.args)
                
                proved, cf, sub_chain = self.backward_chain(cond_goal, visited.copy())
                
                if condition.negated:
                    proved = not proved
                    cf = CertaintyFactor(1 - cf.value)
                
                if not proved:
                    all_proved = False
                    break
                
                condition_cfs.append(cf)
                proof_chain.extend(sub_chain)
            
            if all_proved and condition_cfs:
                # 计算组合确定性因子
                combined_cf = condition_cfs[0]
                for cf in condition_cfs[1:]:
                    combined_cf = combined_cf * cf
                
                final_cf = combined_cf * rule.cf
                
                return True, final_cf, proof_chain
        
        return False, CertaintyFactor(0.0), []
    
    def _resolve_conflicts(self, triggered_rules: List[Tuple[Rule, Fact, Dict]]) -> Tuple[Rule, Fact, Dict]:
        """冲突消解"""
        if len(triggered_rules) == 1:
            return triggered_rules[0]
        
        if self.conflict_resolution == ConflictResolutionStrategy.PRIORITY:
            # 按优先级选择
            triggered_rules.sort(key=lambda x: x[0].priority, reverse=True)
        
        elif self.conflict_resolution == ConflictResolutionStrategy.RECENCY:
            # 按最近最少使用选择
            triggered_rules.sort(key=lambda x: x[0].usage_count)
        
        elif self.conflict_resolution == ConflictResolutionStrategy.SPECIFICITY:
            # 按条件数量（特殊性）选择
            triggered_rules.sort(
                key=lambda x: len(x[0].conditions),
                reverse=True
            )
        
        elif self.conflict_resolution == ConflictResolutionStrategy.CERTAINTY:
            # 按结论确定性选择
            triggered_rules.sort(key=lambda x: x[1].cf.value, reverse=True)
        
        return triggered_rules[0]
    
    def explain(self, fact: Fact) -> str:
        """
        解释事实是如何被推导出来的
        """
        for log_entry in self.inference_log:
            if log_entry['conclusion'] == fact:
                rule = log_entry['rule']
                bindings = log_entry['bindings']
                explanation = f"{fact} was derived using rule '{rule}'\n"
                explanation += f"  Bindings: {bindings}"
                return explanation
        
        return f"{fact} is a given fact."


class NeuralRuleLearner(nn.Module):
    """
    神经规则学习器
    
    从数据中学习规则及其确定性因子。
    """
    
    def __init__(self, num_predicates: int, num_entities: int, embedding_dim: int = 128):
        super().__init__()
        self.num_predicates = num_predicates
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        
        # 谓词和实体嵌入
        self.predicate_embeddings = nn.Embedding(num_predicates, embedding_dim)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # 规则发现网络
        self.rule_discovery = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_predicates)
        )
        
        # 确定性预测器
        self.certainty_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def discover_rules(self,
                      facts: List[Tuple[int, int, int]],
                      min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """
        从事实中发现规则
        
        Args:
            facts: (subject, predicate, object)三元组列表
            min_confidence: 最小置信度
        
        Returns:
            发现的规则列表
        """
        discovered_rules = []
        
        # 统计共现模式
        predicate_pairs = defaultdict(lambda: defaultdict(int))
        
        # 找到所有两跳路径
        fact_dict = defaultdict(list)
        for s, p, o in facts:
            fact_dict[s].append((p, o))
        
        for subject, relations in fact_dict.items():
            for i, (p1, o1) in enumerate(relations):
                for p2, o2 in relations[i+1:]:
                    # 两个关系共享同一个主体
                    predicate_pairs[p1][p2] += 1
        
        # 生成候选规则
        for p1, targets in predicate_pairs.items():
            total = sum(targets.values())
            for p2, count in targets.items():
                confidence = count / total
                if confidence >= min_confidence:
                    discovered_rules.append({
                        'if': p1,
                        'then': p2,
                        'confidence': confidence,
                        'support': count
                    })
        
        return discovered_rules
    
    def predict_certainty(self,
                         rule_embedding: torch.Tensor,
                         context: torch.Tensor) -> torch.Tensor:
        """预测规则的确定性因子"""
        combined = torch.cat([rule_embedding, context])
        return self.certainty_predictor(combined)


# ==================== 材料科学规则库 ====================

def create_material_rules() -> List[Rule]:
    """创建材料科学领域的规则库"""
    
    rules = [
        # 半导体识别规则
        Rule(
            name="semiconductor_by_bandgap",
            conditions=[
                Condition("hasBandGap", ["?material", "?gap"]),
            ],
            conclusion=Fact("isSemiconductor", ["?material"]),
            cf=CertaintyFactor(0.9),
            priority=10,
            meta={"description": "有带隙的材料是半导体"}
        ),
        
        # 导体识别规则
        Rule(
            name="conductor_by_zero_gap",
            conditions=[
                Condition("hasBandGap", ["?material", 0]),
            ],
            conclusion=Fact("isConductor", ["?material"]),
            cf=CertaintyFactor(0.95),
            priority=10
        ),
        
        # 绝缘体识别规则
        Rule(
            name="insulator_by_large_gap",
            conditions=[
                Condition("hasBandGap", ["?material", "?gap"]),
                Condition("greater_than", ["?gap", 3.0]),
            ],
            conclusion=Fact("isInsulator", ["?material"]),
            cf=CertaintyFactor(0.85),
            priority=9
        ),
        
        # 高导电材料
        Rule(
            name="high_conductivity",
            conditions=[
                Condition("hasConductivity", ["?material", "?cond"]),
                Condition("greater_than", ["?cond", 1000]),
            ],
            conclusion=Fact("isGoodConductor", ["?material"]),
            cf=CertaintyFactor(0.9),
            priority=8
        ),
        
        # 二维材料特性
        Rule(
            name="2d_material_properties",
            conditions=[
                Condition("hasDimensionality", ["?material", 2]),
                Condition("hasLayerCount", ["?material", "?layers"]),
                Condition("less_than", ["?layers", 10]),
            ],
            conclusion=Fact("is2DMaterial", ["?material"]),
            cf=CertaintyFactor(0.95),
            priority=10
        ),
        
        # 拓扑材料推断
        Rule(
            name="topological_insulator",
            conditions=[
                Condition("hasBandGap", ["?material", "?gap"]),
                Condition("less_than", ["?gap", 0.3]),
                Condition("hasSpinOrbitCoupling", ["?material", True]),
            ],
            conclusion=Fact("isTopological", ["?material"]),
            cf=CertaintyFactor(0.7),
            priority=5
        ),
        
        # 热电材料推断
        Rule(
            name="thermoelectric_candidate",
            conditions=[
                Condition("hasSeebeckCoefficient", ["?material", "?seebeck"]),
                Condition("hasElectricalConductivity", ["?material", "?sigma"]),
                Condition("hasThermalConductivity", ["?material", "?kappa"]),
            ],
            conclusion=Fact("isThermoelectric", ["?material"]),
            cf=CertaintyFactor(0.6),
            priority=3
        ),
    ]
    
    return rules


if __name__ == "__main__":
    print("=" * 60)
    print("规则引擎测试")
    print("=" * 60)
    
    # 创建规则引擎
    engine = RuleEngine(conflict_resolution=ConflictResolutionStrategy.PRIORITY)
    
    # 添加规则
    rules = create_material_rules()
    for rule in rules:
        engine.add_rule(rule)
    
    print(f"\n添加了 {len(rules)} 条规则")
    
    # 添加事实
    facts = [
        Fact("hasBandGap", ["silicon"], CertaintyFactor(1.0)),
        Fact("hasBandGap", ["silicon"], CertaintyFactor(1.0)),  # 重复添加
        Fact("hasBandGap", ["copper"], CertaintyFactor(0.0)),
        Fact("hasConductivity", ["copper"], CertaintyFactor(10000)),
    ]
    
    for fact in facts:
        engine.add_fact(fact)
    
    print(f"\n添加了 {len(engine.facts)} 个唯一事实")
    
    # 前向推理
    print("\n前向链式推理:")
    engine.forward_chain()
    
    print(f"推理后事实数量: {len(engine.facts)}")
    for fact in engine.facts:
        print(f"  {fact}")
    
    # 后向推理
    print("\n后向链式推理:")
    goal = Fact("isSemiconductor", ["?x"])
    success, cf, proof = engine.backward_chain(goal)
    print(f"目标: {goal}")
    print(f"成功: {success}, 确定性: {cf}")
    print(f"证明链: {proof}")
    
    # 查询
    print("\n查询测试:")
    results = engine.query(Fact("isSemiconductor", ["?x"]))
    print(f"查询半导体材料: {results}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
