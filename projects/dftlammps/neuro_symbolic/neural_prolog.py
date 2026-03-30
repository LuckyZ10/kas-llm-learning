"""
神经Prolog模块 - 可微分逻辑编程与神经定理证明

结合神经网络的连续优化能力与符号逻辑的可解释性，
实现端到端可微的逻辑推理系统。
"""

from typing import List, Dict, Tuple, Optional, Callable, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import re
from abc import ABC, abstractmethod


class LogicOperator(Enum):
    """逻辑运算符枚举"""
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    FORALL = auto()
    EXISTS = auto()


@dataclass
class Term:
    """逻辑项（常量、变量或函数）"""
    name: str
    is_variable: bool = False
    args: List['Term'] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.name, self.is_variable, tuple(self.args)))
    
    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        return (self.name == other.name and 
                self.is_variable == other.is_variable and
                self.args == other.args)
    
    def __repr__(self):
        if self.args:
            return f"{self.name}({', '.join(str(arg) for arg in self.args)})"
        return self.name


@dataclass
class Atom:
    """逻辑原子（谓词）"""
    predicate: str
    args: List[Term]
    negated: bool = False
    
    def __hash__(self):
        return hash((self.predicate, tuple(self.args), self.negated))
    
    def __eq__(self, other):
        if not isinstance(other, Atom):
            return False
        return (self.predicate == other.predicate and 
                self.args == other.args and
                self.negated == other.negated)
    
    def __repr__(self):
        args_str = ', '.join(str(arg) for arg in self.args)
        result = f"{self.predicate}({args_str})"
        return f"¬{result}" if self.negated else result


@dataclass
class Rule:
    """Horn子句规则：head :- body1, body2, ..."""
    head: Atom
    body: List[Atom] = field(default_factory=list)
    confidence: float = 1.0
    
    def __repr__(self):
        if not self.body:
            return f"{self.head}."
        body_str = ', '.join(str(atom) for atom in self.body)
        return f"{self.head} :- {body_str}."


@dataclass
class Fact:
    """事实（无体部的规则）"""
    atom: Atom
    probability: float = 1.0
    source: Optional[str] = None
    
    def __repr__(self):
        prob_str = f" [{self.probability:.3f}]" if self.probability < 1.0 else ""
        return f"{self.atom}{prob_str}."


class Substitution:
    """变量替换（合一）"""
    
    def __init__(self, mapping: Optional[Dict[str, Term]] = None):
        self.mapping = mapping or {}
    
    def apply(self, term: Term) -> Term:
        """应用替换到项"""
        if term.is_variable and term.name in self.mapping:
            return self.mapping[term.name]
        if term.args:
            new_args = [self.apply(arg) for arg in term.args]
            return Term(term.name, term.is_variable, new_args)
        return term
    
    def apply_to_atom(self, atom: Atom) -> Atom:
        """应用替换到原子"""
        new_args = [self.apply(arg) for arg in atom.args]
        return Atom(atom.predicate, new_args, atom.negated)
    
    def compose(self, other: 'Substitution') -> 'Substitution':
        """组合两个替换"""
        new_mapping = {var: other.apply(term) 
                      for var, term in self.mapping.items()}
        new_mapping.update(other.mapping)
        return Substitution(new_mapping)
    
    def __repr__(self):
        return f"{{{', '.join(f'{k}={v}' for k, v in self.mapping.items())}}}"


class NeuralUnification(nn.Module):
    """
    神经合一模块：使用神经网络学习项之间的相似度
    实现可微分的合一操作
    """
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 符号嵌入层
        self.symbol_embedding = nn.Embedding(10000, embedding_dim)
        
        # 合一网络：计算两个项的合一概率
        self.unification_net = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 结构编码器
        self.structure_encoder = nn.LSTM(
            embedding_dim, embedding_dim // 2, 
            batch_first=True, bidirectional=True
        )
    
    def term_to_embedding(self, term: Term, symbol_dict: Dict[str, int]) -> torch.Tensor:
        """将项转换为嵌入向量"""
        if term.name in symbol_dict:
            idx = symbol_dict[term.name]
        else:
            idx = hash(term.name) % 10000
        
        emb = self.symbol_embedding(torch.tensor(idx))
        
        # 如果是函数，编码参数
        if term.args:
            arg_embs = torch.stack([
                self.term_to_embedding(arg, symbol_dict) 
                for arg in term.args
            ])
            _, (h_n, _) = self.structure_encoder(arg_embs.unsqueeze(0))
            struct_emb = h_n.view(-1)
            return torch.cat([emb, struct_emb[:self.embedding_dim]])
        
        return torch.cat([emb, torch.zeros(self.embedding_dim)])
    
    def forward(self, term1: Term, term2: Term, 
                symbol_dict: Dict[str, int]) -> Tuple[torch.Tensor, Optional[Substitution]]:
        """
        计算两个项的合一概率和替换
        
        Returns:
            probability: 合一概率
            substitution: 最一般合一（如果概率>0.5）
        """
        emb1 = self.term_to_embedding(term1, symbol_dict)
        emb2 = self.term_to_embedding(term2, symbol_dict)
        
        # 计算特征组合
        concat = torch.cat([
            emb1, emb2,
            emb1 * emb2,  # 逐元素乘积
            torch.abs(emb1 - emb2)  # 差值
        ])
        
        prob = self.unification_net(concat)
        
        # 如果可以合一，生成替换
        subst = None
        if prob > 0.5:
            subst = self._compute_mgu(term1, term2)
        
        return prob, subst
    
    def _compute_mgu(self, term1: Term, term2: Term) -> Optional[Substitution]:
        """计算最一般合一（递归算法）"""
        # 情况1：term1是变量
        if term1.is_variable:
            if term1 == term2:
                return Substitution()
            return Substitution({term1.name: term2})
        
        # 情况2：term2是变量
        if term2.is_variable:
            return Substitution({term2.name: term1})
        
        # 情况3：都是常量
        if not term1.args and not term2.args:
            if term1.name == term2.name:
                return Substitution()
            return None
        
        # 情况4：都是函数
        if term1.name != term2.name or len(term1.args) != len(term2.args):
            return None
        
        subst = Substitution()
        for arg1, arg2 in zip(term1.args, term2.args):
            arg_subst = self._compute_mgu(
                subst.apply(arg1), 
                subst.apply(arg2)
            )
            if arg_subst is None:
                return None
            subst = subst.compose(arg_subst)
        
        return subst


class NeuralTheoremProver(nn.Module):
    """
    神经定理证明器
    
    使用神经网络引导的证明搜索，结合可微分的前向和后向链式推理。
    """
    
    def __init__(self, embedding_dim: int = 128, num_hops: int = 5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_hops = num_hops
        
        # 合一模块
        self.unifier = NeuralUnification(embedding_dim)
        
        # 规则选择网络
        self.rule_selector = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 证明路径编码器（图神经网络风格）
        self.proof_encoder = nn.ModuleList([
            nn.Linear(embedding_dim * 2, embedding_dim)
            for _ in range(num_hops)
        ])
        
        # 成功概率预测器
        self.success_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def encode_atom(self, atom: Atom, symbol_dict: Dict[str, int]) -> torch.Tensor:
        """将原子编码为向量"""
        predicate_idx = symbol_dict.get(atom.predicate, 
                                       hash(atom.predicate) % 10000)
        pred_emb = self.unifier.symbol_embedding(torch.tensor(predicate_idx))
        
        # 编码参数
        if atom.args:
            arg_embs = torch.stack([
                self.unifier.term_to_embedding(arg, symbol_dict)
                for arg in atom.args
            ])
            arg_mean = arg_embs.mean(dim=0)[:self.embedding_dim]
        else:
            arg_mean = torch.zeros(self.embedding_dim)
        
        return torch.cat([pred_emb, arg_mean])
    
    def forward_chain_reasoning(self, 
                                 facts: List[Fact],
                                 rules: List[Rule],
                                 query: Atom,
                                 symbol_dict: Dict[str, int],
                                 max_iterations: int = 10) -> torch.Tensor:
        """
        前向链式推理
        
        从已知事实出发，应用规则推导新事实，直到回答查询或达到最大迭代次数。
        """
        # 初始化已知事实集
        known_facts = {(fact.atom, fact.probability) for fact in facts}
        
        for iteration in range(max_iterations):
            new_facts = set()
            
            for rule in rules:
                # 尝试找到与规则体匹配的事实组合
                substitutions = self._match_rule_body(
                    rule, known_facts, symbol_dict
                )
                
                for subst, body_prob in substitutions:
                    # 应用替换到规则头
                    new_head = subst.apply_to_atom(rule.head)
                    
                    # 计算新事实的概率
                    head_prob = body_prob * rule.confidence
                    
                    # 检查是否与查询合一
                    unification_prob, _ = self.unifier(
                        new_head.args[0] if new_head.args else Term(""),
                        query.args[0] if query.args else Term(""),
                        symbol_dict
                    )
                    
                    if unification_prob > 0.7:
                        return torch.tensor(head_prob * unification_prob.item())
                    
                    new_facts.add((new_head, head_prob))
            
            # 添加新事实
            known_facts.update(new_facts)
            
            # 如果没有新事实，停止
            if not new_facts:
                break
        
        # 未找到证明
        return torch.tensor(0.0)
    
    def backward_chain_reasoning(self,
                                  facts: List[Fact],
                                  rules: List[Rule],
                                  query: Atom,
                                  symbol_dict: Dict[str, int],
                                  max_depth: int = 10) -> torch.Tensor:
        """
        后向链式推理
        
        从查询出发，反向应用规则，寻找支持查询的事实。
        """
        return self._backward_chain_recursive(
            query, facts, rules, symbol_dict, max_depth, set()
        )
    
    def _backward_chain_recursive(self,
                                   goal: Atom,
                                   facts: List[Fact],
                                   rules: List[Rule],
                                   symbol_dict: Dict[str, int],
                                   depth: int,
                                   visited: Set[Atom]) -> torch.Tensor:
        """后向链式推理的递归实现"""
        if depth == 0:
            return torch.tensor(0.0)
        
        if goal in visited:
            return torch.tensor(0.0)  # 避免循环
        
        visited.add(goal)
        
        # 检查事实
        max_fact_prob = 0.0
        for fact in facts:
            prob, subst = self.unifier(
                goal.args[0] if goal.args else Term(""),
                fact.atom.args[0] if fact.atom.args else Term(""),
                symbol_dict
            )
            if prob.item() > 0.8:
                max_fact_prob = max(max_fact_prob, fact.probability * prob.item())
        
        if max_fact_prob > 0.9:
            return torch.tensor(max_fact_prob)
        
        # 尝试用规则证明
        rule_probs = []
        for rule in rules:
            # 尝试合一规则头和目标
            unif_prob, subst = self.unifier(
                rule.head.args[0] if rule.head.args else Term(""),
                goal.args[0] if goal.args else Term(""),
                symbol_dict
            )
            
            if unif_prob > 0.5 and subst:
                # 递归证明规则体
                body_probs = []
                for body_atom in rule.body:
                    applied_atom = subst.apply_to_atom(body_atom)
                    body_prob = self._backward_chain_recursive(
                        applied_atom, facts, rules, symbol_dict, 
                        depth - 1, visited.copy()
                    )
                    body_probs.append(body_prob)
                
                if body_probs:
                    # 使用模糊逻辑AND计算体部概率
                    body_and = torch.prod(torch.stack(body_probs))
                    rule_prob = unif_prob * body_and * rule.confidence
                    rule_probs.append(rule_prob)
        
        # 使用模糊逻辑OR组合所有证明路径
        if rule_probs:
            # Soft OR: 1 - prod(1 - p_i)
            probs_tensor = torch.stack(rule_probs)
            soft_or = 1 - torch.prod(1 - probs_tensor)
            return torch.maximum(torch.tensor(max_fact_prob), soft_or)
        
        return torch.tensor(max_fact_prob)
    
    def _match_rule_body(self, rule: Rule, facts: Set[Tuple[Atom, float]],
                         symbol_dict: Dict[str, int]) -> List[Tuple[Substitution, float]]:
        """匹配规则体与已知事实"""
        if not rule.body:
            return [(Substitution(), 1.0)]
        
        # 简化实现：尝试匹配每个体部原子
        results = []
        
        # 为每个体部原子收集可能的替换
        body_substitutions = []
        for body_atom in rule.body:
            atom_substs = []
            for fact_atom, fact_prob in facts:
                prob, subst = self.unifier(
                    body_atom.args[0] if body_atom.args else Term(""),
                    fact_atom.args[0] if fact_atom.args else Term(""),
                    symbol_dict
                )
                if prob > 0.7:
                    atom_substs.append((subst, fact_prob * prob.item()))
            body_substitutions.append(atom_substs)
        
        # 组合所有体部原子的替换（笛卡尔积的简化）
        if body_substitutions and all(body_substitutions):
            # 取第一个匹配的替换（简化处理）
            combined_subst = Substitution()
            combined_prob = 1.0
            for substs in body_substitutions:
                if substs:
                    subst, prob = substs[0]
                    combined_subst = combined_subst.compose(subst)
                    combined_prob *= prob
            results.append((combined_subst, combined_prob))
        
        return results


class KnowledgeGraphReasoner(nn.Module):
    """
    知识图谱推理模块
    
    在知识图谱上进行神经符号推理，支持多跳推理和路径查找。
    """
    
    def __init__(self, num_entities: int, num_relations: int, 
                 embedding_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # 实体和关系嵌入
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        
        # 图卷积层用于邻居聚合
        self.gcn_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim)
            for _ in range(num_layers)
        ])
        
        # 路径组合网络
        self.path_combiner = nn.LSTM(
            embedding_dim * 3,  # 实体1 + 关系 + 实体2
            embedding_dim,
            batch_first=True
        )
        
        # 查询编码器
        self.query_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 评分函数（TransE风格）
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, query_triples: torch.Tensor, 
                kg_edges: torch.Tensor) -> torch.Tensor:
        """
        知识图谱推理前向传播
        
        Args:
            query_triples: (batch_size, 3) - (head, relation, tail)查询
            kg_edges: (num_edges, 3) - 知识图谱边
        
        Returns:
            scores: (batch_size,) - 查询的合理性评分
        """
        batch_size = query_triples.shape[0]
        
        # 编码查询
        query_embs = self._encode_triples(query_triples)
        
        # 多跳推理
        for layer in self.gcn_layers:
            query_embs = F.relu(layer(query_embs))
        
        # 计算查询合理性
        scores = self.scorer(query_embs).squeeze(-1)
        
        return torch.sigmoid(scores)
    
    def _encode_triples(self, triples: torch.Tensor) -> torch.Tensor:
        """编码三元组为向量"""
        heads = self.entity_embedding(triples[:, 0])
        relations = self.relation_embedding(triples[:, 1])
        tails = self.entity_embedding(triples[:, 2])
        
        # TransE风格: h + r ≈ t
        return heads + relations - tails
    
    def multi_hop_reasoning(self, 
                           start_entity: int,
                           query_relation: int,
                           kg_edges: torch.Tensor,
                           max_hops: int = 3) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        多跳推理：从起始实体出发，通过多跳关系到达目标实体
        
        Returns:
            entity_scores: 每个实体的可达性分数
            paths: 到达各实体的路径列表
        """
        # 使用BFS风格的神经搜索
        visited = {start_entity: ([start_entity], 1.0)}
        frontier = [(start_entity, [start_entity], 1.0)]
        
        for hop in range(max_hops):
            new_frontier = []
            
            for current_entity, path, prob in frontier:
                # 找到当前实体的所有出边
                outgoing = kg_edges[kg_edges[:, 0] == current_entity]
                
                for edge in outgoing:
                    _, relation, next_entity = edge.tolist()
                    
                    # 计算转移概率
                    rel_sim = F.cosine_similarity(
                        self.relation_embedding(torch.tensor(query_relation)).unsqueeze(0),
                        self.relation_embedding(torch.tensor(relation)).unsqueeze(0),
                        dim=1
                    )
                    
                    new_prob = prob * (0.5 + 0.5 * rel_sim.item())
                    new_path = path + [next_entity]
                    
                    if next_entity not in visited or visited[next_entity][1] < new_prob:
                        visited[next_entity] = (new_path, new_prob)
                        new_frontier.append((next_entity, new_path, new_prob))
            
            frontier = new_frontier
            if not frontier:
                break
        
        # 构建结果
        entity_scores = torch.zeros(self.num_entities)
        paths = [[] for _ in range(self.num_entities)]
        
        for entity, (path, prob) in visited.items():
            entity_scores[entity] = prob
            paths[entity] = path
        
        return entity_scores, paths
    
    def path_ranking(self, 
                    head: int,
                    tail: int,
                    kg_edges: torch.Tensor,
                    max_path_length: int = 4) -> Tuple[torch.Tensor, List[int]]:
        """
        路径排序算法：找到连接两个实体的最合理路径
        """
        # 使用神经引导的A*搜索
        import heapq
        
        open_set = [(0, head, [head])]
        g_score = {head: 0}
        
        head_emb = self.entity_embedding(torch.tensor(head))
        tail_emb = self.entity_embedding(torch.tensor(tail))
        
        while open_set:
            _, current, path = heapq.heappop(open_set)
            
            if current == tail:
                # 将路径转换为关系序列
                path_edges = []
                for i in range(len(path) - 1):
                    edge_mask = ((kg_edges[:, 0] == path[i]) & 
                                (kg_edges[:, 2] == path[i + 1]))
                    if edge_mask.any():
                        rel = kg_edges[edge_mask][0, 1].item()
                        path_edges.append(rel)
                
                # 评分路径
                path_score = self._score_path(path_edges, head_emb, tail_emb)
                return path_score, path_edges
            
            if len(path) >= max_path_length:
                continue
            
            # 扩展邻居
            outgoing = kg_edges[kg_edges[:, 0] == current]
            for edge in outgoing:
                _, relation, next_entity = edge.tolist()
                
                tentative_g = g_score[current] + 1
                
                if next_entity not in g_score or tentative_g < g_score[next_entity]:
                    g_score[next_entity] = tentative_g
                    
                    # 启发式：实体嵌入的余弦相似度
                    next_emb = self.entity_embedding(torch.tensor(next_entity))
                    h = -F.cosine_similarity(next_emb.unsqueeze(0), 
                                            tail_emb.unsqueeze(0), dim=1).item()
                    f = tentative_g + h
                    
                    heapq.heappush(open_set, (f, next_entity, path + [next_entity]))
        
        return torch.tensor(0.0), []
    
    def _score_path(self, relations: List[int], 
                   head_emb: torch.Tensor, 
                   tail_emb: torch.Tensor) -> torch.Tensor:
        """评分路径的合理性"""
        if not relations:
            return torch.tensor(0.0)
        
        # 编码路径
        rel_embs = torch.stack([
            self.relation_embedding(torch.tensor(r)) for r in relations
        ])
        
        # LSTM编码路径
        path_input = torch.cat([
            head_emb.unsqueeze(0).expand(len(relations), -1),
            rel_embs,
            tail_emb.unsqueeze(0).expand(len(relations), -1)
        ], dim=1).unsqueeze(0)
        
        _, (h_n, _) = self.path_combiner(path_input)
        path_emb = h_n.squeeze()
        
        # 计算路径与目标匹配的分数
        target_diff = tail_emb - head_emb
        score = F.cosine_similarity(path_emb.unsqueeze(0), 
                                   target_diff.unsqueeze(0), dim=1)
        
        return torch.sigmoid(score)
    
    def rule_learning(self, 
                     kg_edges: torch.Tensor,
                     min_confidence: float = 0.7) -> List[Rule]:
        """
        从知识图谱中自动学习Horn规则
        
        例如：如果 (A, father, B) 和 (B, father, C) 存在，
             则可能有 (A, grandfather, C)
        """
        learned_rules = []
        
        # 统计关系共现
        relation_paths = defaultdict(lambda: defaultdict(int))
        
        # 找到所有两跳路径
        heads = kg_edges[:, 0].unique()
        for head in heads:
            first_hop = kg_edges[kg_edges[:, 0] == head]
            for edge1 in first_hop:
                _, r1, mid = edge1.tolist()
                second_hop = kg_edges[kg_edges[:, 0] == mid]
                for edge2 in second_hop:
                    _, r2, tail = edge2.tolist()
                    
                    # 检查是否存在直接关系
                    direct = kg_edges[
                        (kg_edges[:, 0] == head) & 
                        (kg_edges[:, 2] == tail)
                    ]
                    for d in direct:
                        r_direct = d[1].item()
                        path_key = (r1, r2)
                        relation_paths[path_key][r_direct] += 1
        
        # 生成规则
        for (r1, r2), targets in relation_paths.items():
            total = sum(targets.values())
            for r_target, count in targets.items():
                confidence = count / total
                if confidence >= min_confidence:
                    # 创建规则
                    head = Atom("related", [
                        Term("A"), Term("C", is_variable=True)
                    ])
                    body = [
                        Atom(f"rel_{r1}", [
                            Term("A"), Term("B", is_variable=True)
                        ]),
                        Atom(f"rel_{r2}", [
                            Term("B", is_variable=True), Term("C", is_variable=True)
                        ])
                    ]
                    rule = Rule(head, body, confidence)
                    learned_rules.append(rule)
        
        return learned_rules


class DifferentiableLogicLayer(nn.Module):
    """
    可微分逻辑层
    
    实现模糊逻辑运算的可微分版本，允许梯度流过逻辑推理。
    """
    
    def __init__(self, logic_type: str = "product"):
        super().__init__()
        self.logic_type = logic_type
        
        # 可学习的逻辑参数
        self.and_params = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.or_params = nn.Parameter(torch.tensor([1.0, 1.0]))
    
    def fuzzy_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """可微分AND运算"""
        if self.logic_type == "product":
            return a * b
        elif self.logic_type == "godel":
            return torch.minimum(a, b)
        elif self.logic_type == "lukasiewicz":
            return torch.maximum(torch.zeros_like(a), a + b - 1)
        elif self.logic_type == "learned":
            # 加权AND
            weights = F.softmax(self.and_params, dim=0)
            return (a ** weights[0]) * (b ** weights[1])
        else:
            return a * b
    
    def fuzzy_or(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """可微分OR运算"""
        if self.logic_type == "product":
            return a + b - a * b
        elif self.logic_type == "godel":
            return torch.maximum(a, b)
        elif self.logic_type == "lukasiewicz":
            return torch.minimum(torch.ones_like(a), a + b)
        elif self.logic_type == "learned":
            weights = F.softmax(self.or_params, dim=0)
            return 1 - (1 - a) ** weights[0] * (1 - b) ** weights[1]
        else:
            return a + b - a * b
    
    def fuzzy_not(self, a: torch.Tensor) -> torch.Tensor:
        """可微分NOT运算"""
        return 1 - a
    
    def fuzzy_implies(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """可微分蕴含运算"""
        # 使用Łukasiewicz蕴含: min(1, 1 - a + b)
        return torch.minimum(torch.ones_like(a), 1 - a + b)
    
    def forward(self, 
                premises: torch.Tensor, 
                operation: str = "and") -> torch.Tensor:
        """
        批量逻辑运算
        
        Args:
            premises: (batch_size, num_premises) 概率值
            operation: "and", "or", "not"
        """
        if operation == "and":
            result = premises[:, 0]
            for i in range(1, premises.shape[1]):
                result = self.fuzzy_and(result, premises[:, i])
            return result
        
        elif operation == "or":
            result = premises[:, 0]
            for i in range(1, premises.shape[1]):
                result = self.fuzzy_or(result, premises[:, i])
            return result
        
        elif operation == "not":
            return self.fuzzy_not(premises)
        
        else:
            raise ValueError(f"Unknown operation: {operation}")


class NeuralLogicProgram(nn.Module):
    """
    神经逻辑程序
    
    将逻辑程序编译为神经网络，实现端到端可微的推理。
    """
    
    def __init__(self, 
                 facts: List[Fact],
                 rules: List[Rule],
                 embedding_dim: int = 128):
        super().__init__()
        self.facts = facts
        self.rules = rules
        self.embedding_dim = embedding_dim
        
        # 可微分逻辑层
        self.logic_layer = DifferentiableLogicLayer(logic_type="learned")
        
        # 事实置信度（可学习）
        self.fact_weights = nn.Parameter(
            torch.tensor([f.probability for f in facts])
        )
        
        # 规则置信度（可学习）
        self.rule_weights = nn.Parameter(
            torch.tensor([r.confidence for r in rules])
        )
        
        # 推理网络
        self.inference_net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query: Atom, symbol_dict: Dict[str, int]) -> torch.Tensor:
        """
        执行神经逻辑推理
        
        将查询编译为神经网络计算图，输出查询成立的概率。
        """
        # 收集支持查询的所有证明路径
        proof_probs = []
        
        # 直接事实匹配
        for i, fact in enumerate(self.facts):
            if self._atoms_match(fact.atom, query):
                proof_probs.append(F.sigmoid(self.fact_weights[i]))
        
        # 规则推理
        for i, rule in enumerate(self.rules):
            if self._atoms_match(rule.head, query):
                # 计算规则体成立的概率
                body_probs = []
                for body_atom in rule.body:
                    # 递归计算体部原子的概率
                    body_prob = self.forward(body_atom, symbol_dict)
                    body_probs.append(body_prob)
                
                if body_probs:
                    body_tensor = torch.stack(body_probs).unsqueeze(0)
                    body_and = self.logic_layer(body_tensor, "and")
                    rule_prob = body_and * F.sigmoid(self.rule_weights[i])
                    proof_probs.append(rule_prob)
        
        if proof_probs:
            # 使用OR组合所有证明路径
            proofs_tensor = torch.stack(proof_probs).unsqueeze(0)
            return self.logic_layer(proofs_tensor, "or")
        
        return torch.tensor(0.0)
    
    def _atoms_match(self, atom1: Atom, atom2: Atom) -> bool:
        """检查两个原子是否匹配（考虑变量）"""
        if atom1.predicate != atom2.predicate:
            return False
        if len(atom1.args) != len(atom2.args):
            return False
        
        for a1, a2 in zip(atom1.args, atom2.args):
            # 如果任一参数是变量，可以匹配
            if a1.is_variable or a2.is_variable:
                continue
            if a1.name != a2.name:
                return False
        
        return True


# ==================== 实用工具和示例 ====================

def parse_term(term_str: str) -> Term:
    """从字符串解析逻辑项"""
    term_str = term_str.strip()
    
    # 检查是否是变量（大写字母开头）
    is_var = term_str[0].isupper() if term_str else False
    
    # 检查是否是函数
    if '(' in term_str and term_str.endswith(')'):
        name = term_str[:term_str.index('(')]
        args_str = term_str[term_str.index('(') + 1:-1]
        args = [parse_term(arg.strip()) for arg in args_str.split(',')]
        return Term(name, is_var, args)
    
    return Term(term_str, is_var)


def parse_atom(atom_str: str) -> Atom:
    """从字符串解析原子"""
    atom_str = atom_str.strip()
    negated = atom_str.startswith('¬') or atom_str.startswith('not ')
    
    if negated:
        atom_str = atom_str[1:] if atom_str.startswith('¬') else atom_str[4:]
    
    if '(' in atom_str and atom_str.endswith(')'):
        predicate = atom_str[:atom_str.index('(')]
        args_str = atom_str[atom_str.index('(') + 1:-1]
        args = [parse_term(arg.strip()) for arg in args_str.split(',')]
        return Atom(predicate, args, negated)
    
    return Atom(atom_str, [], negated)


def parse_rule(rule_str: str) -> Rule:
    """从字符串解析规则"""
    rule_str = rule_str.strip()
    
    if ':-' in rule_str:
        head_str, body_str = rule_str.split(':-')
        head = parse_atom(head_str.strip())
        body_atoms = [parse_atom(a.strip()) 
                     for a in body_str.split('.')[0].split(',')]
        return Rule(head, body_atoms)
    else:
        # 事实是head :- true的简写
        return Rule(parse_atom(rule_str.replace('.', '')))


# 示例：材料知识推理
def create_material_knowledge_base():
    """创建材料科学示例知识库"""
    facts = [
        Fact(parse_atom("crystal_structure(silicon, diamond)"), 1.0),
        Fact(parse_atom("band_gap(silicon, 1.12)"), 1.0),
        Fact(parse_atom("crystal_structure(germanium, diamond)"), 1.0),
        Fact(parse_atom("band_gap(germanium, 0.67)"), 1.0),
        Fact(parse_atom("group(silicon, 14)"), 1.0),
        Fact(parse_atom("group(germanium, 14)"), 1.0),
        Fact(parse_atom("period(silicon, 3)"), 1.0),
        Fact(parse_atom("period(germanium, 4)"), 1.0),
        Fact(parse_atom("conductivity(copper, high)"), 1.0),
        Fact(parse_atom("crystal_structure(copper, fcc)"), 1.0),
    ]
    
    rules = [
        # 如果元素在同一组且有相同的晶体结构，可能有相似性质
        parse_rule("similar_properties(X, Y) :- "
                  "group(X, G), group(Y, G), "
                  "crystal_structure(X, S), crystal_structure(Y, S), "
                  "X \\= Y"),
        
        # 半导体规则
        parse_rule("semiconductor(M) :- "
                  "band_gap(M, G), G > 0.1, G < 4.0"),
        
        # 导体规则
        parse_rule("conductor(M) :- "
                  "conductivity(M, high)"),
    ]
    
    return facts, rules


if __name__ == "__main__":
    # 测试神经Prolog功能
    print("=" * 60)
    print("神经Prolog测试")
    print("=" * 60)
    
    # 创建知识库
    facts, rules = create_material_knowledge_base()
    
    print("\n事实:")
    for fact in facts:
        print(f"  {fact}")
    
    print("\n规则:")
    for rule in rules:
        print(f"  {rule}")
    
    # 创建证明器
    prover = NeuralTheoremProver(embedding_dim=64)
    
    # 符号字典
    symbol_dict = {}
    for fact in facts:
        for arg in fact.atom.args:
            if arg.name not in symbol_dict:
                symbol_dict[arg.name] = len(symbol_dict)
    
    print("\n符号字典:", symbol_dict)
    
    # 测试查询
    query = parse_atom("semiconductor(silicon)")
    print(f"\n查询: {query}")
    
    # 后向链式推理
    result = prover.backward_chain_reasoning(facts, rules, query, symbol_dict)
    print(f"后向链式推理结果: {result.item():.4f}")
    
    # 测试合一
    unifier = NeuralUnification(embedding_dim=64)
    term1 = parse_term("X")
    term2 = parse_term("silicon")
    prob, subst = unifier(term1, term2, symbol_dict)
    print(f"\n合一测试: {term1} 与 {term2}")
    print(f"  概率: {prob.item():.4f}")
    print(f"  替换: {subst}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
