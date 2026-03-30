"""
符号推理引擎 - Symbolic Reasoning Engine
实现逻辑规则推理、知识图谱推理和符号操作
"""

from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
import re
from enum import Enum, auto
import json
import copy


class TermType(Enum):
    """项类型"""
    CONSTANT = auto()
    VARIABLE = auto()
    FUNCTION = auto()
    PREDICATE = auto()


@dataclass
class Term:
    """逻辑项"""
    name: str
    term_type: TermType
    args: List['Term'] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.name, self.term_type, tuple(self.args)))
    
    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        return (self.name == other.name and 
                self.term_type == other.term_type and
                self.args == other.args)
    
    def __repr__(self):
        if self.args:
            args_str = ", ".join(str(arg) for arg in self.args)
            return f"{self.name}({args_str})"
        return self.name
    
    def is_variable(self) -> bool:
        return self.term_type == TermType.VARIABLE
    
    def is_constant(self) -> bool:
        return self.term_type == TermType.CONSTANT
    
    def is_ground(self) -> bool:
        """检查是否为基项（不含变量）"""
        if self.is_variable():
            return False
        return all(arg.is_ground() for arg in self.args)
    
    def variables(self) -> Set[str]:
        """获取所有变量名"""
        if self.is_variable():
            return {self.name}
        result = set()
        for arg in self.args:
            result.update(arg.variables())
        return result
    
    def substitute(self, substitution: Dict[str, 'Term']) -> 'Term':
        """应用替换"""
        if self.is_variable() and self.name in substitution:
            return substitution[self.name]
        new_args = [arg.substitute(substitution) for arg in self.args]
        return Term(self.name, self.term_type, new_args)


@dataclass
class Literal:
    """文字（谓词原子）"""
    predicate: str
    args: List[Term]
    negated: bool = False
    
    def __hash__(self):
        return hash((self.predicate, tuple(self.args), self.negated))
    
    def __eq__(self, other):
        if not isinstance(other, Literal):
            return False
        return (self.predicate == other.predicate and 
                self.args == other.args and
                self.negated == other.negated)
    
    def __repr__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        lit = f"{self.predicate}({args_str})"
        return f"¬{lit}" if self.negated else lit
    
    def negate(self) -> 'Literal':
        """取反"""
        return Literal(self.predicate, self.args, not self.negated)
    
    def is_ground(self) -> bool:
        return all(arg.is_ground() for arg in self.args)
    
    def variables(self) -> Set[str]:
        result = set()
        for arg in self.args:
            result.update(arg.variables())
        return result
    
    def substitute(self, substitution: Dict[str, Term]) -> 'Literal':
        new_args = [arg.substitute(substitution) for arg in self.args]
        return Literal(self.predicate, new_args, self.negated)


@dataclass
class Rule:
    """逻辑规则 (Horn子句)"""
    head: Literal
    body: List[Literal] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.head, tuple(self.body)))
    
    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return self.head == other.head and self.body == other.body
    
    def __repr__(self):
        if self.body:
            body_str = " ∧ ".join(str(lit) for lit in self.body)
            return f"{self.head} ← {body_str}"
        return str(self.head)
    
    def is_fact(self) -> bool:
        """是否为事实（无体部）"""
        return len(self.body) == 0
    
    def variables(self) -> Set[str]:
        result = self.head.variables()
        for lit in self.body:
            result.update(lit.variables())
        return result
    
    def substitute(self, substitution: Dict[str, Term]) -> 'Rule':
        new_head = self.head.substitute(substitution)
        new_body = [lit.substitute(substitution) for lit in self.body]
        return Rule(new_head, new_body, self.metadata)


class KnowledgeBase:
    """
    知识库
    存储和管理逻辑规则与事实
    """
    
    def __init__(self):
        self.rules: List[Rule] = []
        self.facts: Set[Literal] = set()
        self.predicates: Dict[str, List[Rule]] = defaultdict(list)
        self.index: Dict[str, Set[Literal]] = defaultdict(set)
    
    def add_rule(self, rule: Rule):
        """添加规则"""
        self.rules.append(rule)
        self.predicates[rule.head.predicate].append(rule)
    
    def add_fact(self, fact: Literal):
        """添加事实"""
        if not fact.negated:
            self.facts.add(fact)
            self.index[fact.predicate].add(fact)
    
    def add_facts(self, facts: List[Literal]):
        """批量添加事实"""
        for fact in facts:
            self.add_fact(fact)
    
    def query(self, literal: Literal) -> List[Dict[str, Term]]:
        """
        查询满足文字的所有替换
        
        Returns:
            使文字为真的所有变量替换列表
        """
        results = []
        
        # 检查事实
        for fact in self.index[literal.predicate]:
            unifier = unify(literal, fact)
            if unifier is not None:
                results.append(unifier)
        
        # 尝试用规则推导
        for rule in self.predicates[literal.predicate]:
            unifier = unify(literal, rule.head)
            if unifier is not None:
                # 这里简化处理，实际需要递归求解
                results.append(unifier)
        
        return results
    
    def get_rules_for(self, predicate: str) -> List[Rule]:
        """获取关于某谓词的所有规则"""
        return self.predicates[predicate]
    
    def get_facts_for(self, predicate: str) -> Set[Literal]:
        """获取关于某谓词的所有事实"""
        return self.index[predicate]
    
    def to_string(self) -> str:
        """转换为字符串表示"""
        lines = []
        lines.append("=== Facts ===")
        for fact in sorted(self.facts, key=str):
            lines.append(f"  {fact}")
        lines.append("\n=== Rules ===")
        for rule in self.rules:
            lines.append(f"  {rule}")
        return "\n".join(lines)


def unify(t1: Union[Term, Literal], t2: Union[Term, Literal]) -> Optional[Dict[str, Term]]:
    """
    最一般合一算法 (MGU)
    
    Args:
        t1, t2: 要合一的项或文字
    
    Returns:
        最一般合一替换，或None（如果不能合一）
    """
    substitution = {}
    
    def unify_terms(a: Term, b: Term, subst: Dict[str, Term]) -> Optional[Dict[str, Term]]:
        a = a.substitute(subst)
        b = b.substitute(subst)
        
        if a == b:
            return subst
        
        if a.is_variable():
            if occurs_in(a.name, b):
                return None
            new_subst = subst.copy()
            new_subst[a.name] = b
            return new_subst
        
        if b.is_variable():
            if occurs_in(b.name, a):
                return None
            new_subst = subst.copy()
            new_subst[b.name] = a
            return new_subst
        
        if a.term_type != b.term_type or a.name != b.name:
            return None
        
        if len(a.args) != len(b.args):
            return None
        
        current_subst = subst.copy()
        for arg_a, arg_b in zip(a.args, b.args):
            result = unify_terms(arg_a, arg_b, current_subst)
            if result is None:
                return None
            current_subst = result
        
        return current_subst
    
    def occurs_in(var: str, term: Term) -> bool:
        """检查变量是否出现在项中（Occurs检查）"""
        if term.is_variable():
            return term.name == var
        return any(occurs_in(var, arg) for arg in term.args)
    
    if isinstance(t1, Literal) and isinstance(t2, Literal):
        if t1.predicate != t2.predicate or t1.negated != t2.negated:
            return None
        if len(t1.args) != len(t2.args):
            return None
        
        current_subst = {}
        for arg1, arg2 in zip(t1.args, t2.args):
            result = unify_terms(arg1, arg2, current_subst)
            if result is None:
                return None
            current_subst = result
        
        return current_subst
    
    elif isinstance(t1, Term) and isinstance(t2, Term):
        return unify_terms(t1, t2, {})
    
    return None


class SLDResolution:
    """
    SLD归结推理
    Prolog风格的反向链接推理
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.max_depth = 100
        self.proof_trace: List[Dict] = []
    
    def prove(
        self,
        goal: Literal,
        max_solutions: int = 10
    ) -> List[Dict[str, Term]]:
        """
        证明目标，返回所有解
        
        Args:
            goal: 要证明的目标
            max_solutions: 最大解的数量
        
        Returns:
            所有使目标为真的变量替换
        """
        solutions = []
        self.proof_trace = []
        
        self._prove_recursive(
            [goal],
            {},
            solutions,
            max_solutions,
            depth=0
        )
        
        return solutions
    
    def _prove_recursive(
        self,
        goals: List[Literal],
        substitution: Dict[str, Term],
        solutions: List[Dict[str, Term]],
        max_solutions: int,
        depth: int
    ) -> bool:
        """递归证明"""
        if depth > self.max_depth:
            return False
        
        if len(solutions) >= max_solutions:
            return True
        
        if not goals:
            # 成功！所有子目标都已证明
            solutions.append(substitution)
            return True
        
        # 选择第一个子目标
        current_goal = goals[0].substitute(substitution)
        remaining_goals = goals[1:]
        
        # 尝试所有匹配的规则和事实
        found_solution = False
        
        # 尝试事实
        for fact in self.kb.get_facts_for(current_goal.predicate):
            unifier = unify(current_goal, fact)
            if unifier is not None:
                new_substitution = self._compose_substitutions(substitution, unifier)
                found = self._prove_recursive(
                    remaining_goals,
                    new_substitution,
                    solutions,
                    max_solutions,
                    depth + 1
                )
                found_solution = found_solution or found
        
        # 尝试规则
        for rule in self.kb.get_rules_for(current_goal.predicate):
            renamed_rule = self._rename_variables(rule, depth)
            unifier = unify(current_goal, renamed_rule.head)
            if unifier is not None:
                new_goals = [lit.substitute(unifier) for lit in renamed_rule.body] + remaining_goals
                new_substitution = self._compose_substitutions(substitution, unifier)
                found = self._prove_recursive(
                    new_goals,
                    new_substitution,
                    solutions,
                    max_solutions,
                    depth + 1
                )
                found_solution = found_solution or found
        
        return found_solution
    
    def _compose_substitutions(
        self,
        s1: Dict[str, Term],
        s2: Dict[str, Term]
    ) -> Dict[str, Term]:
        """组合两个替换"""
        result = {}
        # 应用s1到s2的值
        for var, term in s2.items():
            result[var] = term.substitute(s1)
        # 添加s1中不在s2中的替换
        for var, term in s1.items():
            if var not in result:
                result[var] = term
        return result
    
    def _rename_variables(self, rule: Rule, depth: int) -> Rule:
        """重命名变量以避免冲突"""
        vars = rule.variables()
        rename_map = {}
        for var in vars:
            rename_map[var] = Term(f"{var}_{depth}", TermType.VARIABLE)
        return rule.substitute(rename_map)


@dataclass
class Node:
    """知识图谱节点"""
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id


@dataclass
class Edge:
    """知识图谱边"""
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.source, self.target, self.relation))
    
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.source == other.source and 
                self.target == other.target and
                self.relation == other.relation)


class KnowledgeGraph:
    """
    知识图谱
    存储实体和关系，支持图推理
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Set[Edge] = set()
        self.outgoing: Dict[str, List[Edge]] = defaultdict(list)
        self.incoming: Dict[str, List[Edge]] = defaultdict(list)
        self.relation_index: Dict[str, List[Edge]] = defaultdict(list)
    
    def add_node(self, node_id: str, label: str, **properties) -> Node:
        """添加节点"""
        node = Node(node_id, label, properties)
        self.nodes[node_id] = node
        return node
    
    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        **properties
    ) -> Edge:
        """添加边"""
        edge = Edge(source, target, relation, properties)
        self.edges.add(edge)
        self.outgoing[source].append(edge)
        self.incoming[target].append(edge)
        self.relation_index[relation].append(edge)
        return edge
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def get_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None
    ) -> List[Node]:
        """获取邻居节点"""
        edges = self.outgoing[node_id]
        if relation:
            edges = [e for e in edges if e.relation == relation]
        return [self.nodes[e.target] for e in edges if e.target in self.nodes]
    
    def get_related(
        self,
        node_id: str,
        relation: str
    ) -> List[Node]:
        """获取通过特定关系关联的节点"""
        edges = [e for e in self.outgoing[node_id] if e.relation == relation]
        return [self.nodes[e.target] for e in edges if e.target in self.nodes]
    
    def find_paths(
        self,
        source: str,
        target: str,
        max_length: int = 5
    ) -> List[List[str]]:
        """
        查找两个节点间的所有路径
        
        Args:
            source: 源节点ID
            target: 目标节点ID
            max_length: 最大路径长度
        
        Returns:
            所有路径（节点ID列表）
        """
        paths = []
        visited = set()
        
        def dfs(current: str, path: List[str]):
            if len(path) > max_length:
                return
            if current == target:
                paths.append(path.copy())
                return
            
            visited.add(current)
            for edge in self.outgoing[current]:
                if edge.target not in visited:
                    path.append(edge.target)
                    dfs(edge.target, path)
                    path.pop()
            visited.remove(current)
        
        dfs(source, [source])
        return paths
    
    def transitive_closure(
        self,
        relation: str,
        node_id: str
    ) -> Set[str]:
        """
        计算传递闭包
        
        Args:
            relation: 关系类型
            node_id: 起始节点
        
        Returns:
        可到达的所有节点ID
        """
        visited = set()
        stack = [node_id]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for edge in self.outgoing[current]:
                if edge.relation == relation and edge.target not in visited:
                    stack.append(edge.target)
        
        visited.remove(node_id)
        return visited
    
    def query_pattern(
        self,
        pattern: List[Tuple[str, str, str]]
    ) -> List[Dict[str, str]]:
        """
        模式查询
        
        Args:
            pattern: [(变量1, 关系, 变量2), ...]
        
        Returns:
            所有匹配的变量绑定
        """
        if not pattern:
            return [{}]
        
        results = []
        self._query_pattern_recursive(pattern, {}, 0, results)
        return results
    
    def _query_pattern_recursive(
        self,
        pattern: List[Tuple[str, str, str]],
        binding: Dict[str, str],
        idx: int,
        results: List[Dict[str, str]]
    ):
        """递归模式查询"""
        if idx >= len(pattern):
            results.append(binding.copy())
            return
        
        var1, relation, var2 = pattern[idx]
        
        # 确定起始节点
        if var1 in binding:
            start_nodes = [binding[var1]]
        else:
            start_nodes = list(self.nodes.keys())
        
        for start in start_nodes:
            for edge in self.outgoing[start]:
                if edge.relation == relation:
                    target = edge.target
                    
                    # 检查一致性
                    new_binding = binding.copy()
                    if var1 not in new_binding:
                        new_binding[var1] = start
                    elif new_binding[var1] != start:
                        continue
                    
                    if var2 not in new_binding:
                        new_binding[var2] = target
                    elif new_binding[var2] != target:
                        continue
                    
                    self._query_pattern_recursive(
                        pattern, new_binding, idx + 1, results
                    )
    
    def to_cypher(self) -> str:
        """转换为Cypher查询语句（用于Neo4j）"""
        statements = []
        
        # 创建节点
        for node in self.nodes.values():
            props = ", ".join([f"{k}: '{v}'" for k, v in node.properties.items()])
            if props:
                props = f" {{{props}}}"
            statements.append(
                f"CREATE (:{node.label} {{id: '{node.id}'{props}}})"
            )
        
        # 创建关系
        for edge in self.edges:
            props = ", ".join([f"{k}: '{v}'" for k, v in edge.properties.items()])
            if props:
                props = f" {{{props}}}"
            statements.append(
                f"MATCH (a {{id: '{edge.source}'}}), (b {{id: '{edge.target}'}}) "
                f"CREATE (a)-[:{edge.relation}{props}]->(b)"
            )
        
        return "\n".join(statements)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "properties": n.properties
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "properties": e.properties
                }
                for e in self.edges
            ]
        }


class RuleMiner:
    """
    规则挖掘器
    从知识图谱中自动挖掘规则
    """
    
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
    
    def mine_association_rules(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5
    ) -> List[Rule]:
        """
        挖掘关联规则
        
        Args:
            min_support: 最小支持度
            min_confidence: 最小置信度
        
        Returns:
            挖掘出的规则列表
        """
        rules = []
        
        # 统计关系共现
        relation_pairs = defaultdict(int)
        relation_counts = defaultdict(int)
        total = 0
        
        for node in self.kg.nodes.values():
            outgoing = [e.relation for e in self.kg.outgoing[node.id]]
            relation_counts.update(outgoing)
            
            for r1, r2 in itertools.combinations(outgoing, 2):
                relation_pairs[(r1, r2)] += 1
                total += 1
        
        # 生成规则
        for (r1, r2), count in relation_pairs.items():
            support = count / total
            if support >= min_support:
                confidence = count / relation_counts[r1]
                if confidence >= min_confidence:
                    # 创建规则: r1(X,Y) -> r2(X,Y)
                    x = Term("X", TermType.VARIABLE)
                    y = Term("Y", TermType.VARIABLE)
                    
                    head = Literal(r2, [x, y])
                    body = [Literal(r1, [x, y])]
                    
                    rule = Rule(head, body, {
                        "support": support,
                        "confidence": confidence
                    })
                    rules.append(rule)
        
        return rules
    
    def mine_transitive_rules(self) -> List[Rule]:
        """挖掘传递性规则"""
        rules = []
        
        for relation in self.kg.relation_index.keys():
            # 检查传递性: r(X,Y) ∧ r(Y,Z) -> r(X,Z)
            violations = 0
            total = 0
            
            edges = self.kg.relation_index[relation]
            for e1 in edges:
                for e2 in edges:
                    if e1.target == e2.source:
                        total += 1
                        # 检查是否存在传递边
                        exists = any(
                            e.source == e1.source and e.target == e2.target
                            for e in edges
                        )
                        if not exists:
                            violations += 1
            
            if total > 0 and violations / total < 0.1:
                # 具有传递性
                x = Term("X", TermType.VARIABLE)
                y = Term("Y", TermType.VARIABLE)
                z = Term("Z", TermType.VARIABLE)
                
                head = Literal(relation, [x, z])
                body = [
                    Literal(relation, [x, y]),
                    Literal(relation, [y, z])
                ]
                
                rule = Rule(head, body, {
                    "type": "transitive",
                    "accuracy": 1 - violations / total
                })
                rules.append(rule)
        
        return rules


class SymbolicReasoner:
    """
    符号推理器主类
    整合逻辑推理和知识图谱推理
    """
    
    def __init__(self):
        self.kb = KnowledgeBase()
        self.kg = KnowledgeGraph()
        self.sld = SLDResolution(self.kb)
        self.miner = RuleMiner(self.kg)
    
    def add_rule(self, rule: Rule):
        """添加逻辑规则"""
        self.kb.add_rule(rule)
    
    def add_fact(self, fact: Literal):
        """添加事实"""
        self.kb.add_fact(fact)
    
    def add_knowledge_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        **properties
    ):
        """添加知识三元组"""
        # 确保节点存在
        if subject not in self.kg.nodes:
            self.kg.add_node(subject, "Entity")
        if object not in self.kg.nodes:
            self.kg.add_node(object, "Entity")
        
        self.kg.add_edge(subject, object, predicate, **properties)
        
        # 同时添加到逻辑知识库
        subj_term = Term(subject, TermType.CONSTANT)
        obj_term = Term(object, TermType.CONSTANT)
        fact = Literal(predicate, [subj_term, obj_term])
        self.kb.add_fact(fact)
    
    def query(
        self,
        query_str: str
    ) -> List[Dict[str, Any]]:
        """
        通用查询接口
        
        Args:
            query_str: 查询字符串，如 "parent(X, Y)"
        
        Returns:
            查询结果
        """
        # 解析查询
        literal = self._parse_literal(query_str)
        
        # 尝试逻辑推理
        solutions = self.sld.prove(literal)
        
        # 转换结果为字典格式
        results = []
        for sol in solutions:
            result = {}
            for var, term in sol.items():
                result[var] = term.name if term.is_constant() else str(term)
            results.append(result)
        
        return results
    
    def _parse_literal(self, query_str: str) -> Literal:
        """解析查询字符串"""
        # 简化的解析器
        match = re.match(r"(\w+)\(([^)]+)\)", query_str.strip())
        if match:
            predicate = match.group(1)
            args_str = match.group(2)
            args = []
            for arg in args_str.split(","):
                arg = arg.strip()
                if arg[0].isupper():
                    args.append(Term(arg, TermType.VARIABLE))
                else:
                    args.append(Term(arg, TermType.CONSTANT))
            return Literal(predicate, args)
        
        raise ValueError(f"无法解析查询: {query_str}")
    
    def explain(self, fact: Literal) -> Optional[List[Rule]]:
        """
        解释一个事实是如何被推导出来的
        
        Args:
            fact: 要解释的事实
        
        Returns:
            推导该事实的规则链
        """
        # 简化实现：查找可以推导出该事实的规则
        rules = self.kb.get_rules_for(fact.predicate)
        return rules if rules else None
    
    def infer_new_facts(self) -> List[Literal]:
        """推理新事实"""
        new_facts = []
        
        # 应用所有规则
        for rule in self.kb.rules:
            if rule.is_fact():
                continue
            
            # 这里简化处理，实际需要合一和替换
            # 检查是否可以满足体部
            # 如果可以，添加头部为事实
            pass  # 简化实现
        
        return new_facts
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "facts": len(self.kb.facts),
            "rules": len(self.kb.rules),
            "nodes": len(self.kg.nodes),
            "edges": len(self.kg.edges)
        }


def demo():
    """演示符号推理引擎"""
    print("=" * 60)
    print("符号推理引擎演示")
    print("=" * 60)
    
    # 创建推理器
    reasoner = SymbolicReasoner()
    
    print("\n1. 构建知识库")
    # 添加关于家族关系的知识
    family_data = [
        ("john", "father", "paul"),
        ("paul", "father", "mary"),
        ("mary", "mother", "alice"),
        ("susan", "mother", "paul"),
        ("tom", "father", "john"),
    ]
    
    for subj, pred, obj in family_data:
        reasoner.add_knowledge_triple(subj, pred, obj)
        print(f"   添加: {subj} --{pred}--> {obj}")
    
    # 添加推理规则
    x = Term("X", TermType.VARIABLE)
    y = Term("Y", TermType.VARIABLE)
    z = Term("Z", TermType.VARIABLE)
    
    # 祖父规则
    grandfather_rule = Rule(
        head=Literal("grandfather", [x, z]),
        body=[
            Literal("father", [x, y]),
            Literal("father", [y, z])
        ],
        metadata={"description": "祖父定义"}
    )
    reasoner.add_rule(grandfather_rule)
    print(f"\n   添加规则: {grandfather_rule}")
    
    print("\n2. 知识图谱查询")
    # 查找John的所有后代
    descendants = reasoner.kg.transitive_closure("father", "john")
    print(f"   John的后代: {descendants}")
    
    # 查找路径
    paths = reasoner.kg.find_paths("tom", "alice", max_length=5)
    print(f"   Tom到Alice的路径数: {len(paths)}")
    for i, path in enumerate(paths[:3], 1):
        print(f"     路径{i}: {' -> '.join(path)}")
    
    print("\n3. 规则挖掘")
    rules = reasoner.miner.mine_association_rules(min_support=0.1, min_confidence=0.5)
    print(f"   挖掘到 {len(rules)} 条关联规则")
    for rule in rules[:3]:
        print(f"     {rule}")
        print(f"       支持度: {rule.metadata.get('support', 0):.3f}")
        print(f"       置信度: {rule.metadata.get('confidence', 0):.3f}")
    
    print("\n4. 逻辑推理")
    # 查询祖父关系
    results = reasoner.query("grandfather(X, Y)")
    print(f"   查询 grandfather(X, Y):")
    for result in results:
        print(f"     {result}")
    
    print("\n5. 统计信息")
    stats = reasoner.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
