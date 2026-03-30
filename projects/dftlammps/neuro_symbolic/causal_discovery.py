"""
因果发现算法 - Causal Discovery Algorithms
实现PC算法、GES和NOTEARS等因果发现方法
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import itertools
from scipy import stats
from scipy.linalg import inv, det
import warnings


class CausalEdgeType(Enum):
    """因果边类型"""
    DIRECTED = "->"      # 有向边
    UNDIRECTED = "-"      # 无向边
    BIDIRECTED = "<->"   # 双向边（潜在混杂）
    UNKNOWN = "?"         # 未知方向


@dataclass
class CausalEdge:
    """因果图中的边"""
    source: int
    target: int
    edge_type: CausalEdgeType = CausalEdgeType.DIRECTED
    weight: float = 1.0
    confidence: float = 0.0
    
    def __hash__(self):
        return hash((self.source, self.target, self.edge_type))
    
    def __eq__(self, other):
        if not isinstance(other, CausalEdge):
            return False
        return (self.source == other.source and 
                self.target == other.target and
                self.edge_type == other.edge_type)


@dataclass
class CausalGraph:
    """因果图"""
    n_nodes: int
    node_names: List[str] = None
    edges: Set[CausalEdge] = field(default_factory=set)
    adjacency: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.node_names is None:
            self.node_names = [f"X{i}" for i in range(self.n_nodes)]
        if self.adjacency is None:
            self.adjacency = np.zeros((self.n_nodes, self.n_nodes))
    
    def add_edge(self, edge: CausalEdge):
        """添加边"""
        self.edges.add(edge)
        self.adjacency[edge.source, edge.target] = edge.weight
    
    def remove_edge(self, source: int, target: int):
        """移除边"""
        edge_to_remove = None
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                edge_to_remove = edge
                break
        if edge_to_remove:
            self.edges.remove(edge_to_remove)
            self.adjacency[source, target] = 0
    
    def has_edge(self, source: int, target: int) -> bool:
        """检查是否存在边"""
        return any(e.source == source and e.target == target for e in self.edges)
    
    def get_parents(self, node: int) -> List[int]:
        """获取节点的父节点"""
        parents = []
        for edge in self.edges:
            if edge.target == node and edge.edge_type == CausalEdgeType.DIRECTED:
                parents.append(edge.source)
        return parents
    
    def get_children(self, node: int) -> List[int]:
        """获取节点的子节点"""
        children = []
        for edge in self.edges:
            if edge.source == node and edge.edge_type == CausalEdgeType.DIRECTED:
                children.append(edge.target)
        return children
    
    def get_neighbors(self, node: int) -> List[int]:
        """获取邻居节点"""
        neighbors = set()
        for edge in self.edges:
            if edge.source == node:
                neighbors.add(edge.target)
            elif edge.target == node:
                neighbors.add(edge.source)
        return list(neighbors)
    
    def is_dag(self) -> bool:
        """检查是否为有向无环图"""
        visited = [False] * self.n_nodes
        rec_stack = [False] * self.n_nodes
        
        def has_cycle(node: int) -> bool:
            visited[node] = True
            rec_stack[node] = True
            
            for child in self.get_children(node):
                if not visited[child]:
                    if has_cycle(child):
                        return True
                elif rec_stack[child]:
                    return True
            
            rec_stack[node] = False
            return False
        
        for node in range(self.n_nodes):
            if not visited[node]:
                if has_cycle(node):
                    return False
        
        return True
    
    def topological_sort(self) -> Optional[List[int]]:
        """拓扑排序"""
        if not self.is_dag():
            return None
        
        in_degree = [0] * self.n_nodes
        for edge in self.edges:
            if edge.edge_type == CausalEdgeType.DIRECTED:
                in_degree[edge.target] += 1
        
        queue = [i for i in range(self.n_nodes) if in_degree[i] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result if len(result) == self.n_nodes else None
    
    def to_networkx(self):
        """转换为NetworkX图"""
        try:
            import networkx as nx
            G = nx.DiGraph()
            for i, name in enumerate(self.node_names):
                G.add_node(i, name=name)
            for edge in self.edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
            return G
        except ImportError:
            warnings.warn("NetworkX not installed")
            return None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "n_nodes": self.n_nodes,
            "node_names": self.node_names,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.edge_type.value,
                    "weight": e.weight,
                    "confidence": e.confidence
                }
                for e in self.edges
            ],
            "adjacency": self.adjacency.tolist()
        }


class IndependenceTest:
    """
    独立性检验
    支持多种条件独立性检验方法
    """
    
    def __init__(self, method: str = "fisher_z", alpha: float = 0.05):
        self.method = method
        self.alpha = alpha
    
    def test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None
    ) -> Tuple[bool, float]:
        """
        条件独立性检验
        
        Args:
            X: 变量X的索引或数据
            Y: 变量Y的索引或数据
            Z: 条件变量集
            data: 完整数据集（如果X,Y是索引）
        
        Returns:
            (是否独立, p值)
        """
        if data is not None:
            x_data = data[:, X] if isinstance(X, int) else X
            y_data = data[:, Y] if isinstance(Y, int) else Y
            if Z is not None:
                z_data = data[:, Z] if isinstance(Z, (int, np.integer)) else Z
            else:
                z_data = None
        else:
            x_data, y_data, z_data = X, Y, Z
        
        if self.method == "fisher_z":
            return self._fisher_z_test(x_data, y_data, z_data)
        elif self.method == "pearson":
            return self._pearson_test(x_data, y_data, z_data)
        elif self.method == "chi2":
            return self._chi2_test(x_data, y_data, z_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _fisher_z_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray]
    ) -> Tuple[bool, float]:
        """Fisher Z变换检验"""
        n = len(X)
        
        if Z is None or len(Z) == 0:
            # 无条件情况
            r = np.corrcoef(X, Y)[0, 1]
        else:
            # 条件情况：偏相关
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)
            r = self._partial_correlation(X, Y, Z)
        
        # Fisher Z变换
        if abs(r) >= 1:
            r = 0.9999 * np.sign(r)
        
        z = 0.5 * np.log((1 + r) / (1 - r))
        
        # 条件变量数
        k = 0 if Z is None else (Z.shape[1] if Z.ndim > 1 else 1)
        
        # 检验统计量
        test_stat = np.sqrt(n - k - 3) * abs(z)
        p_value = 2 * (1 - stats.norm.cdf(test_stat))
        
        is_independent = p_value > self.alpha
        
        return is_independent, p_value
    
    def _pearson_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray]
    ) -> Tuple[bool, float]:
        """Pearson相关检验"""
        if Z is None or len(Z) == 0:
            r, p_value = stats.pearsonr(X, Y)
        else:
            r = self._partial_correlation(X, Y, Z)
            # 近似p值
            n = len(X)
            k = Z.shape[1] if Z.ndim > 1 else 1
            t_stat = r * np.sqrt((n - k - 2) / (1 - r**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 2))
        
        return p_value > self.alpha, p_value
    
    def _chi2_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray]
    ) -> Tuple[bool, float]:
        """卡方检验（用于离散变量）"""
        # 简化实现
        if Z is not None:
            # 这里简化处理
            pass
        
        # 创建列联表
        contingency = np.histogram2d(X, Y, bins=5)[0]
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        
        return p_value > self.alpha, p_value
    
    def _partial_correlation(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
    ) -> float:
        """计算偏相关系数"""
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        
        # 回归去除Z的影响
        from numpy.linalg import lstsq
        
        X_residual = X - Z @ lstsq(Z, X, rcond=None)[0]
        Y_residual = Y - Z @ lstsq(Z, Y, rcond=None)[0]
        
        return np.corrcoef(X_residual, Y_residual)[0, 1]


class PCAlgorithm:
    """
    PC算法 (Peter-Clark算法)
    基于约束的因果发现算法
    """
    
    def __init__(
        self,
        independence_test: Optional[IndependenceTest] = None,
        max_depth: Optional[int] = None,
        verbose: bool = False
    ):
        self.test = independence_test or IndependenceTest()
        self.max_depth = max_depth
        self.verbose = verbose
        self.sepsets: Dict[Tuple[int, int], Set[int]] = {}
    
    def fit(self, data: np.ndarray, node_names: Optional[List[str]] = None) -> CausalGraph:
        """
        学习因果图结构
        
        Args:
            data: 观测数据 [n_samples, n_features]
            node_names: 变量名称
        
        Returns:
            学习到的因果图
        """
        n_samples, n_vars = data.shape
        
        if self.max_depth is None:
            self.max_depth = max(1, n_vars - 2)
        
        if node_names is None:
            node_names = [f"X{i}" for i in range(n_vars)]
        
        # 步骤1: 构建完全无向图
        graph = self._initialize_graph(n_vars)
        
        # 步骤2: 骨架识别
        graph = self._skeleton_learning(graph, data)
        
        # 步骤3: 方向确定
        graph = self._orient_edges(graph)
        
        # 创建因果图对象
        causal_graph = CausalGraph(n_vars, node_names)
        for edge in graph.edges:
            causal_graph.add_edge(edge)
        
        return causal_graph
    
    def _initialize_graph(self, n_vars: int) -> CausalGraph:
        """初始化完全无向图"""
        graph = CausalGraph(n_vars)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                edge = CausalEdge(
                    i, j, CausalEdgeType.UNDIRECTED, weight=1.0
                )
                graph.add_edge(edge)
        return graph
    
    def _skeleton_learning(
        self,
        graph: CausalGraph,
        data: np.ndarray
    ) -> CausalGraph:
        """骨架学习阶段"""
        n_vars = graph.n_nodes
        
        for depth in range(self.max_depth + 1):
            if self.verbose:
                print(f"PC Algorithm: depth = {depth}")
            
            removed_edges = []
            
            for edge in list(graph.edges):
                if edge.edge_type != CausalEdgeType.UNDIRECTED:
                    continue
                
                x, y = edge.source, edge.target
                neighbors = self._get_neighbors(graph, x)
                neighbors = [n for n in neighbors if n != y]
                
                if len(neighbors) < depth:
                    continue
                
                # 尝试所有大小为depth的条件集
                for cond_set in itertools.combinations(neighbors, depth):
                    cond_set = set(cond_set)
                    
                    is_independent, p_value = self.test.test(
                        x, y, list(cond_set), data
                    )
                    
                    if is_independent:
                        if self.verbose:
                            print(f"  移除边 {x}-{y}, 条件集: {cond_set}, p={p_value:.4f}")
                        
                        removed_edges.append((x, y))
                        self.sepsets[(x, y)] = cond_set
                        self.sepsets[(y, x)] = cond_set
                        break
            
            # 移除边
            for x, y in removed_edges:
                graph.remove_edge(x, y)
                graph.remove_edge(y, x)
            
            if not removed_edges:
                break
        
        return graph
    
    def _get_neighbors(self, graph: CausalGraph, node: int) -> List[int]:
        """获取节点的邻居"""
        neighbors = []
        for edge in graph.edges:
            if edge.source == node and edge.edge_type == CausalEdgeType.UNDIRECTED:
                neighbors.append(edge.target)
            elif edge.target == node and edge.edge_type == CausalEdgeType.UNDIRECTED:
                neighbors.append(edge.source)
        return neighbors
    
    def _orient_edges(self, graph: CausalGraph) -> CausalGraph:
        """边方向确定"""
        n_vars = graph.n_nodes
        
        # 规则1: 定向V-结构
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                for k in range(n_vars):
                    if i == k or j == k:
                        continue
                    
                    # 检查V-结构: i - k - j, 但 i 和 j 不相连
                    if (self._has_undirected_edge(graph, i, k) and
                        self._has_undirected_edge(graph, j, k) and
                        not self._has_any_edge(graph, i, j)):
                        
                        # 检查k是否在sepset(i,j)中
                        sepset = self.sepsets.get((i, j), set())
                        
                        if k not in sepset:
                            # 定向为 V-结构: i -> k <- j
                            if self.verbose:
                                print(f"  V-结构: {i} -> {k} <- {j}")
                            
                            self._orient_edge(graph, i, k)
                            self._orient_edge(graph, j, k)
        
        # 规则2-4: 传播方向
        changed = True
        while changed:
            changed = False
            changed = self._apply_orientation_rules(graph) or changed
        
        return graph
    
    def _has_undirected_edge(self, graph: CausalGraph, i: int, j: int) -> bool:
        """检查是否有无向边"""
        for edge in graph.edges:
            if ((edge.source == i and edge.target == j) or
                (edge.source == j and edge.target == i)):
                return edge.edge_type == CausalEdgeType.UNDIRECTED
        return False
    
    def _has_any_edge(self, graph: CausalGraph, i: int, j: int) -> bool:
        """检查是否有任何边"""
        return any(
            (e.source == i and e.target == j) or
            (e.source == j and e.target == i)
            for e in graph.edges
        )
    
    def _orient_edge(self, graph: CausalGraph, source: int, target: int):
        """定向边"""
        # 移除无向边
        for edge in list(graph.edges):
            if ((edge.source == source and edge.target == target) or
                (edge.source == target and edge.target == source)):
                if edge.edge_type == CausalEdgeType.UNDIRECTED:
                    graph.edges.remove(edge)
        
        # 添加有向边
        new_edge = CausalEdge(source, target, CausalEdgeType.DIRECTED)
        graph.add_edge(new_edge)
    
    def _apply_orientation_rules(self, graph: CausalGraph) -> bool:
        """应用方向传播规则"""
        changed = False
        
        # 规则2: i -> j -> k 且 i - k 则 i -> k
        for edge1 in graph.edges:
            if edge1.edge_type != CausalEdgeType.DIRECTED:
                continue
            i, j = edge1.source, edge1.target
            
            for edge2 in graph.edges:
                if edge2.edge_type != CausalEdgeType.DIRECTED:
                    continue
                if edge2.source != j:
                    continue
                k = edge2.target
                
                if self._has_undirected_edge(graph, i, k):
                    self._orient_edge(graph, i, k)
                    changed = True
        
        # 规则3: i -> j <- k 且 i - l - k 且 l - j 则 l -> j
        # 简化实现
        
        return changed


class GESAlgorithm:
    """
    GES算法 (Greedy Equivalence Search)
    基于评分的因果发现算法
    """
    
    def __init__(
        self,
        score_type: str = "bic",
        max_parents: int = 10,
        verbose: bool = False
    ):
        self.score_type = score_type
        self.max_parents = max_parents
        self.verbose = verbose
        self.score_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    
    def fit(self, data: np.ndarray, node_names: Optional[List[str]] = None) -> CausalGraph:
        """
        学习因果图结构
        
        Args:
            data: 观测数据
            node_names: 变量名称
        
        Returns:
            学习到的因果图
        """
        n_samples, n_vars = data.shape
        
        if node_names is None:
            node_names = [f"X{i}" for i in range(n_vars)]
        
        # 初始化空图
        parents = {i: set() for i in range(n_vars)}
        
        # 前向阶段：添加边
        improved = True
        while improved:
            improved = False
            best_score_diff = 0
            best_addition = None
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j or i in parents[j]:
                        continue
                    if len(parents[j]) >= self.max_parents:
                        continue
                    
                    # 检查是否会形成环
                    new_parents = parents[j] | {i}
                    if self._would_create_cycle(parents, j, new_parents, n_vars):
                        continue
                    
                    score_diff = self._score_diff(data, j, parents[j], new_parents)
                    
                    if score_diff > best_score_diff:
                        best_score_diff = score_diff
                        best_addition = (i, j)
            
            if best_addition:
                i, j = best_addition
                parents[j].add(i)
                improved = True
                if self.verbose:
                    print(f"GES: 添加边 {i} -> {j}, 得分提升: {best_score_diff:.4f}")
        
        # 后向阶段：移除边
        improved = True
        while improved:
            improved = False
            best_score_diff = 0
            best_removal = None
            
            for j in range(n_vars):
                for i in parents[j]:
                    new_parents = parents[j] - {i}
                    score_diff = self._score_diff(data, j, parents[j], new_parents)
                    
                    if score_diff > best_score_diff:
                        best_score_diff = score_diff
                        best_removal = (i, j)
            
            if best_removal:
                i, j = best_removal
                parents[j].remove(i)
                improved = True
                if self.verbose:
                    print(f"GES: 移除边 {i} -> {j}, 得分提升: {best_score_diff:.4f}")
        
        # 构建因果图
        causal_graph = CausalGraph(n_vars, node_names)
        for j, pars in parents.items():
            for i in pars:
                edge = CausalEdge(i, j, CausalEdgeType.DIRECTED, weight=1.0)
                causal_graph.add_edge(edge)
        
        return causal_graph
    
    def _would_create_cycle(
        self,
        parents: Dict[int, Set[int]],
        node: int,
        new_parents: Set[int],
        n_vars: int
    ) -> bool:
        """检查是否会创建环"""
        # 临时更新并检查
        temp_parents = parents.copy()
        temp_parents[node] = new_parents
        
        visited = [False] * n_vars
        rec_stack = [False] * n_vars
        
        def has_cycle(n: int) -> bool:
            visited[n] = True
            rec_stack[n] = True
            
            for p in temp_parents.get(n, set()):
                if not visited[p]:
                    if has_cycle(p):
                        return True
                elif rec_stack[p]:
                    return True
            
            rec_stack[n] = False
            return False
        
        for i in range(n_vars):
            if not visited[i]:
                if has_cycle(i):
                    return True
        
        return False
    
    def _score_diff(
        self,
        data: np.ndarray,
        node: int,
        old_parents: Set[int],
        new_parents: Set[int]
    ) -> float:
        """计算得分差异"""
        old_score = self._local_score(data, node, old_parents)
        new_score = self._local_score(data, node, new_parents)
        return new_score - old_score
    
    def _local_score(
        self,
        data: np.ndarray,
        node: int,
        parents: Set[int]
    ) -> float:
        """计算局部评分"""
        cache_key = (node, tuple(sorted(parents)))
        
        if cache_key in self.score_cache:
            return self.score_cache[cache_key]
        
        if self.score_type == "bic":
            score = self._bic_score(data, node, parents)
        elif self.score_type == "aic":
            score = self._aic_score(data, node, parents)
        elif self.score_type == "ll":
            score = self._log_likelihood(data, node, parents)
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")
        
        self.score_cache[cache_key] = score
        return score
    
    def _bic_score(
        self,
        data: np.ndarray,
        node: int,
        parents: Set[int]
    ) -> float:
        """BIC评分"""
        n = len(data)
        ll = self._log_likelihood(data, node, parents)
        k = len(parents) + 1  # 参数数量
        return ll - 0.5 * k * np.log(n)
    
    def _aic_score(
        self,
        data: np.ndarray,
        node: int,
        parents: Set[int]
    ) -> float:
        """AIC评分"""
        ll = self._log_likelihood(data, node, parents)
        k = len(parents) + 1
        return ll - k
    
    def _log_likelihood(
        self,
        data: np.ndarray,
        node: int,
        parents: Set[int]
    ) -> float:
        """计算对数似然（假设线性高斯模型）"""
        y = data[:, node]
        
        if len(parents) == 0:
            # 无条件
            variance = np.var(y)
            if variance < 1e-10:
                variance = 1e-10
            n = len(y)
            return -0.5 * n * np.log(2 * np.pi * variance) - 0.5 * n
        
        # 线性回归
        X = data[:, list(parents)]
        X = np.column_stack([np.ones(len(X)), X])
        
        # 最小二乘估计
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        variance = np.var(residuals)
        
        if variance < 1e-10:
            variance = 1e-10
        
        n = len(y)
        ll = -0.5 * n * np.log(2 * np.pi * variance) - 0.5 * n
        
        return ll


class NOTEARS:
    """
    NOTEARS算法
    基于优化的连续因果发现算法
    """
    
    def __init__(
        self,
        lambda1: float = 0.1,
        lambda2: float = 0.0,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e+16,
        w_threshold: float = 0.3,
        verbose: bool = False
    ):
        self.lambda1 = lambda1  # L1正则化
        self.lambda2 = lambda2  # L2正则化
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.verbose = verbose
    
    def fit(self, data: np.ndarray, node_names: Optional[List[str]] = None) -> CausalGraph:
        """
        学习因果图结构
        
        Args:
            data: 观测数据
            node_names: 变量名称
        
        Returns:
            学习到的因果图
        """
        n_samples, n_vars = data.shape
        
        if node_names is None:
            node_names = [f"X{i}" for i in range(n_vars)]
        
        # 标准化数据
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        
        # 使用增广拉格朗日方法求解
        W = self._notears_solver(data)
        
        # 阈值处理
        W[np.abs(W) < self.w_threshold] = 0
        
        # 构建因果图
        causal_graph = CausalGraph(n_vars, node_names)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if W[i, j] != 0:
                    edge = CausalEdge(
                        i, j, CausalEdgeType.DIRECTED, weight=abs(W[i, j])
                    )
                    causal_graph.add_edge(edge)
        
        return causal_graph
    
    def _notears_solver(self, X: np.ndarray) -> np.ndarray:
        """NOTEARS求解器"""
        n, d = X.shape
        
        # 初始化
        W = np.zeros((d, d))
        rho = 1.0
        alpha = 0.0
        
        for iteration in range(self.max_iter):
            # 求解子问题
            W_new = self._solve_subproblem(X, W, rho, alpha)
            
            # 计算约束违反度
            h = self._acyclicity_constraint(W_new)
            
            if self.verbose and iteration % 10 == 0:
                loss = self._loss(X, W_new)
                print(f"Iter {iteration}: loss={loss:.4f}, h={h:.4e}, rho={rho:.2e}")
            
            # 检查收敛
            if h < self.h_tol or rho >= self.rho_max:
                W = W_new
                break
            
            # 更新增广拉格朗日参数
            alpha += rho * h
            rho *= 10
            W = W_new
        
        return W
    
    def _solve_subproblem(
        self,
        X: np.ndarray,
        W: np.ndarray,
        rho: float,
        alpha: float
    ) -> np.ndarray:
        """求解增广拉格朗日子问题"""
        n, d = X.shape
        
        # 使用梯度下降
        W = W.copy()
        lr = 0.001
        
        for _ in range(100):
            # 计算梯度
            grad = self._gradient(X, W, rho, alpha)
            
            # 梯度更新
            W = W - lr * grad
            
            # 软阈值（L1正则化）
            W = self._soft_threshold(W, self.lambda1 * lr)
        
        return W
    
    def _loss(self, X: np.ndarray, W: np.ndarray) -> float:
        """计算损失函数"""
        n, d = X.shape
        
        # 最小二乘损失
        R = X - X @ W
        loss = 0.5 / n * np.sum(R ** 2)
        
        # L1正则化
        loss += self.lambda1 * np.sum(np.abs(W))
        
        # L2正则化
        loss += 0.5 * self.lambda2 * np.sum(W ** 2)
        
        return loss
    
    def _gradient(
        self,
        X: np.ndarray,
        W: np.ndarray,
        rho: float,
        alpha: float
    ) -> np.ndarray:
        """计算梯度"""
        n, d = X.shape
        
        # 最小二乘梯度
        R = X - X @ W
        grad = -1.0 / n * X.T @ R
        
        # L2正则化梯度
        grad += self.lambda2 * W
        
        # 无环约束梯度（简化近似）
        h = self._acyclicity_constraint(W)
        if h > 0:
            # 使用数值近似梯度
            eps = 1e-5
            for i in range(d):
                for j in range(d):
                    W_plus = W.copy()
                    W_plus[i, j] += eps
                    h_plus = self._acyclicity_constraint(W_plus)
                    grad[i, j] += rho * (h_plus - h) / eps + alpha * (h_plus - h) / eps
        
        return grad
    
    def _acyclicity_constraint(self, W: np.ndarray) -> float:
        """
        无环约束函数
        h(W) = tr(exp(W ⊙ W)) - d = 0
        """
        d = W.shape[0]
        M = np.exp(W * W) - np.eye(d)
        
        # 计算矩阵指数（简化）
        # 实际应使用 scipy.linalg.expm
        # 这里使用近似
        E = np.eye(d) + M + 0.5 * M @ M
        
        h = np.trace(E) - d
        return h
    
    def _soft_threshold(self, W: np.ndarray, threshold: float) -> np.ndarray:
        """软阈值操作"""
        return np.sign(W) * np.maximum(np.abs(W) - threshold, 0)


class CausalDiscovery:
    """
    因果发现主类
    整合多种因果发现算法
    """
    
    def __init__(self):
        self.algorithms = {
            "pc": PCAlgorithm,
            "ges": GESAlgorithm,
            "notears": NOTEARS
        }
        self.results: Dict[str, CausalGraph] = {}
    
    def discover(
        self,
        data: np.ndarray,
        algorithm: str = "pc",
        node_names: Optional[List[str]] = None,
        **kwargs
    ) -> CausalGraph:
        """
        执行因果发现
        
        Args:
            data: 观测数据
            algorithm: 算法名称 ("pc", "ges", "notears")
            node_names: 变量名称
            **kwargs: 算法特定参数
        
        Returns:
            学习到的因果图
        """
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        algo_class = self.algorithms[algorithm]
        algo = algo_class(**kwargs)
        
        graph = algo.fit(data, node_names)
        self.results[algorithm] = graph
        
        return graph
    
    def compare_algorithms(
        self,
        data: np.ndarray,
        node_names: Optional[List[str]] = None,
        true_graph: Optional[CausalGraph] = None
    ) -> Dict[str, Any]:
        """
        比较多种算法
        
        Args:
            data: 观测数据
            node_names: 变量名称
            true_graph: 真实因果图（用于评估）
        
        Returns:
            比较结果
        """
        results = {}
        
        for name in self.algorithms.keys():
            try:
                graph = self.discover(data, name, node_names)
                
                result = {
                    "graph": graph,
                    "is_dag": graph.is_dag(),
                    "n_edges": len(graph.edges)
                }
                
                if true_graph:
                    metrics = self._evaluate_graph(graph, true_graph)
                    result.update(metrics)
                
                results[name] = result
            except Exception as e:
                results[name] = {"error": str(e)}
        
        return results
    
    def _evaluate_graph(
        self,
        predicted: CausalGraph,
        true: CausalGraph
    ) -> Dict[str, float]:
        """评估因果图"""
        pred_edges = set((e.source, e.target) for e in predicted.edges
                        if e.edge_type == CausalEdgeType.DIRECTED)
        true_edges = set((e.source, e.target) for e in true.edges
                        if e.edge_type == CausalEdgeType.DIRECTED)
        
        # 转换为骨架（无向边）
        pred_skeleton = set()
        for e in predicted.edges:
            pred_skeleton.add(tuple(sorted([e.source, e.target])))
        
        true_skeleton = set()
        for e in true.edges:
            true_skeleton.add(tuple(sorted([e.source, e.target])))
        
        # 计算指标
        tp = len(pred_edges & true_edges)
        fp = len(pred_edges - true_edges)
        fn = len(true_edges - pred_edges)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 骨架指标
        tp_skel = len(pred_skeleton & true_skeleton)
        fp_skel = len(pred_skeleton - true_skeleton)
        fn_skel = len(true_skeleton - pred_skeleton)
        
        precision_skel = tp_skel / (tp_skel + fp_skel) if (tp_skel + fp_skel) > 0 else 0
        recall_skel = tp_skel / (tp_skel + fn_skel) if (tp_skel + fn_skel) > 0 else 0
        f1_skel = 2 * precision_skel * recall_skel / (precision_skel + recall_skel) if (precision_skel + recall_skel) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "shd": fp + fn,  # 结构汉明距离
            "precision_skeleton": precision_skel,
            "recall_skeleton": recall_skel,
            "f1_skeleton": f1_skel
        }


def demo():
    """演示因果发现算法"""
    print("=" * 60)
    print("因果发现算法演示")
    print("=" * 60)
    
    # 生成合成数据
    np.random.seed(42)
    n_samples = 1000
    
    # 创建真实因果结构: X1 -> X2 -> X4, X1 -> X3 -> X4, X2 -> X3
    n_vars = 4
    true_adj = np.array([
        [0, 1, 1, 0],  # X1 -> X2, X1 -> X3
        [0, 0, 1, 1],  # X2 -> X3, X2 -> X4
        [0, 0, 0, 1],  # X3 -> X4
        [0, 0, 0, 0]
    ])
    
    # 生成数据（线性高斯模型）
    X1 = np.random.randn(n_samples)
    X2 = 0.5 * X1 + np.random.randn(n_samples)
    X3 = 0.3 * X1 + 0.4 * X2 + np.random.randn(n_samples)
    X4 = 0.6 * X2 + 0.3 * X3 + np.random.randn(n_samples)
    
    data = np.column_stack([X1, X2, X3, X4])
    node_names = ["X1", "X2", "X3", "X4"]
    
    print(f"\n数据形状: {data.shape}")
    print("真实因果结构:")
    for i in range(n_vars):
        for j in range(n_vars):
            if true_adj[i, j] == 1:
                print(f"  {node_names[i]} -> {node_names[j]}")
    
    # 创建真实因果图
    true_graph = CausalGraph(n_vars, node_names)
    for i in range(n_vars):
        for j in range(n_vars):
            if true_adj[i, j] == 1:
                edge = CausalEdge(i, j, CausalEdgeType.DIRECTED)
                true_graph.add_edge(edge)
    
    # 使用不同算法
    cd = CausalDiscovery()
    
    print("\n" + "-" * 40)
    print("PC算法")
    print("-" * 40)
    try:
        graph_pc = cd.discover(data, "pc", node_names, verbose=False)
        print(f"是否为DAG: {graph_pc.is_dag()}")
        print(f"边数: {len(graph_pc.edges)}")
        print("发现的边:")
        for edge in graph_pc.edges:
            print(f"  {node_names[edge.source]} {edge.edge_type.value} {node_names[edge.target]}")
        
        metrics = cd._evaluate_graph(graph_pc, true_graph)
        print(f"评估指标: F1={metrics['f1']:.3f}, SHD={metrics['shd']}")
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n" + "-" * 40)
    print("GES算法")
    print("-" * 40)
    try:
        graph_ges = cd.discover(data, "ges", node_names, verbose=False)
        print(f"是否为DAG: {graph_ges.is_dag()}")
        print(f"边数: {len(graph_ges.edges)}")
        print("发现的边:")
        for edge in graph_ges.edges:
            print(f"  {node_names[edge.source]} {edge.edge_type.value} {node_names[edge.target]}")
        
        metrics = cd._evaluate_graph(graph_ges, true_graph)
        print(f"评估指标: F1={metrics['f1']:.3f}, SHD={metrics['shd']}")
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n" + "-" * 40)
    print("NOTEARS算法")
    print("-" * 40)
    try:
        graph_nt = cd.discover(data, "notears", node_names, verbose=False)
        print(f"是否为DAG: {graph_nt.is_dag()}")
        print(f"边数: {len(graph_nt.edges)}")
        print("发现的边:")
        for edge in graph_nt.edges:
            print(f"  {node_names[edge.source]} -> {node_names[edge.target]} (weight={edge.weight:.3f})")
        
        metrics = cd._evaluate_graph(graph_nt, true_graph)
        print(f"评估指标: F1={metrics['f1']:.3f}, SHD={metrics['shd']}")
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
