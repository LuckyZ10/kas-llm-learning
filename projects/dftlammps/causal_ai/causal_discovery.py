"""
因果发现模块 - Causal Discovery for Materials

本模块实现材料科学中的因果发现算法，包括：
- 结构学习算法 (PC, GES, NOTEARS)
- 干预效应估计
- 反事实推理

作者: Causal AI Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
import itertools
import networkx as nx
from abc import ABC, abstractmethod


class IndependenceTest(Enum):
    """独立性检验方法"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    PARTIAL_CORRELATION = "partial_correlation"
    FISHER_Z = "fisher_z"
    KERNEL_CI = "kernel_ci"
    HSIC = "hsic"  # Hilbert-Schmidt Independence Criterion


class CausalAlgorithm(Enum):
    """因果发现算法类型"""
    PC = "pc"
    GES = "ges"
    NOTEARS = "notears"
    DAG_GNN = "dag_gnn"
    CAUSAL_FOREST = "causal_forest"


@dataclass
class CausalEdge:
    """因果图中的边"""
    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "directed"  # directed, bidirected, undirected
    confidence: float = 0.0
    mechanism: Optional[str] = None  # 机制描述
    
    def __hash__(self):
        return hash((self.source, self.target, self.edge_type))
    
    def __eq__(self, other):
        if isinstance(other, CausalEdge):
            return (self.source, self.target, self.edge_type) == \
                   (other.source, other.target, other.edge_type)
        return False


@dataclass
class Intervention:
    """干预操作定义"""
    variable: str
    value: float
    intervention_type: str = "do"  # do, shift, noise
    
    def __str__(self):
        return f"do({self.variable}={self.value})"


@dataclass
class CounterfactualQuery:
    """反事实查询"""
    factual_evidence: Dict[str, float]  # 事实证据
    hypothetical_intervention: Intervention  # 假设干预
    target_variable: str  # 目标变量


class CausalGraph:
    """因果图结构"""
    
    def __init__(self, variables: List[str] = None):
        self.variables = variables or []
        self.edges: Set[CausalEdge] = set()
        self.adjacency_matrix: Optional[np.ndarray] = None
        self.var_to_idx: Dict[str, int] = {}
        self.graph = nx.DiGraph()
        self.mechanisms: Dict[str, Callable] = {}
        self.noise_distributions: Dict[str, Callable] = {}
        
    def add_edge(self, edge: CausalEdge):
        """添加边到因果图"""
        self.edges.add(edge)
        self.graph.add_edge(edge.source, edge.target, 
                           weight=edge.weight, 
                           confidence=edge.confidence)
        
    def remove_edge(self, source: str, target: str):
        """移除边"""
        self.edges = {e for e in self.edges 
                     if not (e.source == source and e.target == target)}
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
    
    def get_parents(self, variable: str) -> List[str]:
        """获取变量的父节点"""
        return list(self.graph.predecessors(variable))
    
    def get_children(self, variable: str) -> List[str]:
        """获取变量的子节点"""
        return list(self.graph.successors(variable))
    
    def get_ancestors(self, variable: str) -> Set[str]:
        """获取变量的祖先节点"""
        return nx.ancestors(self.graph, variable)
    
    def get_descendants(self, variable: str) -> Set[str]:
        """获取变量的后代节点"""
        return nx.descendants(self.graph, variable)
    
    def is_d_separated(self, x: str, y: str, conditioning_set: Set[str] = None) -> bool:
        """检查d-分离"""
        if conditioning_set is None:
            conditioning_set = set()
        return nx.d_separated(self.graph, {x}, {y}, conditioning_set)
    
    def get_markov_equivalent_class(self) -> List['CausalGraph']:
        """获取马尔科夫等价类"""
        # 实现CPDAG（完成部分定向图）
        cpdag = self._to_cpdag()
        return [self._from_cpdag(cpdag)]
    
    def _to_cpdag(self) -> nx.Graph:
        """转换为CPDAG"""
        # 简化的CPDAG实现
        cpdag = nx.Graph()
        for edge in self.edges:
            cpdag.add_edge(edge.source, edge.target)
        return cpdag
    
    def _from_cpdag(self, cpdag: nx.Graph) -> 'CausalGraph':
        """从CPDAG重构因果图"""
        graph = CausalGraph(list(cpdag.nodes()))
        for u, v in cpdag.edges():
            graph.add_edge(CausalEdge(u, v))
        return graph
    
    def visualize(self, ax=None, **kwargs):
        """可视化因果图"""
        try:
            import matplotlib.pyplot as plt
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 8))
            
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
            
            # 绘制节点
            node_colors = [kwargs.get('node_color', 'lightblue')] * len(self.graph.nodes())
            nx.draw_networkx_nodes(self.graph, pos, ax=ax, 
                                  node_color=node_colors, 
                                  node_size=kwargs.get('node_size', 1500),
                                  alpha=0.9)
            
            # 绘制边
            edge_colors = [edge.confidence for edge in self.edges]
            nx.draw_networkx_edges(self.graph, pos, ax=ax,
                                  edge_color=edge_colors,
                                  edge_cmap=plt.cm.Reds,
                                  width=2,
                                  arrowsize=20,
                                  arrowstyle='->')
            
            # 绘制标签
            nx.draw_networkx_labels(self.graph, pos, ax=ax, font_size=10)
            
            # 绘制边权重
            edge_labels = {(e.source, e.target): f"{e.weight:.2f}" 
                          for e in self.edges}
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, ax=ax)
            
            ax.set_title(kwargs.get('title', 'Causal Graph'))
            ax.axis('off')
            return ax
        except ImportError:
            warnings.warn("matplotlib not available for visualization")
            return None
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """转换为邻接矩阵"""
        n = len(self.variables)
        adj = np.zeros((n, n))
        for edge in self.edges:
            i = self.variables.index(edge.source)
            j = self.variables.index(edge.target)
            adj[i, j] = edge.weight
        return adj
    
    def from_adjacency_matrix(self, adj: np.ndarray, variables: List[str]):
        """从邻接矩阵构建因果图"""
        self.variables = variables
        self.edges = set()
        n = len(variables)
        for i in range(n):
            for j in range(n):
                if abs(adj[i, j]) > 1e-6:
                    self.add_edge(CausalEdge(
                        variables[i], variables[j], 
                        weight=adj[i, j]
                    ))


class IndependenceTester:
    """独立性检验器"""
    
    def __init__(self, test_type: IndependenceTest = IndependenceTest.PEARSON,
                 alpha: float = 0.05):
        self.test_type = test_type
        self.alpha = alpha
        
    def test(self, x: np.ndarray, y: np.ndarray, 
             z: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        检验条件独立性
        
        Returns:
            (是否独立, p值)
        """
        if self.test_type == IndependenceTest.PEARSON:
            return self._pearson_test(x, y, z)
        elif self.test_type == IndependenceTest.SPEARMAN:
            return self._spearman_test(x, y, z)
        elif self.test_type == IndependenceTest.PARTIAL_CORRELATION:
            return self._partial_correlation_test(x, y, z)
        elif self.test_type == IndependenceTest.FISHER_Z:
            return self._fisher_z_test(x, y, z)
        elif self.test_type == IndependenceTest.HSIC:
            return self._hsic_test(x, y, z)
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")
    
    def _pearson_test(self, x: np.ndarray, y: np.ndarray,
                      z: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """Pearson相关检验"""
        if z is not None:
            # 残差化
            x_res = self._residualize(x, z)
            y_res = self._residualize(y, z)
            corr, pvalue = pearsonr(x_res, y_res)
        else:
            corr, pvalue = pearsonr(x, y)
        return pvalue > self.alpha, pvalue
    
    def _spearman_test(self, x: np.ndarray, y: np.ndarray,
                       z: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """Spearman秩相关检验"""
        if z is not None:
            x_res = self._residualize(x, z)
            y_res = self._residualize(y, z)
            corr, pvalue = spearmanr(x_res, y_res)
        else:
            corr, pvalue = spearmanr(x, y)
        return pvalue > self.alpha, pvalue
    
    def _partial_correlation_test(self, x: np.ndarray, y: np.ndarray,
                                   z: np.ndarray) -> Tuple[bool, float]:
        """偏相关检验"""
        # 计算偏相关系数
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        
        # 残差化
        x_res = self._residualize(x, z)
        y_res = self._residualize(y, z)
        
        corr, pvalue = pearsonr(x_res, y_res)
        return pvalue > self.alpha, pvalue
    
    def _fisher_z_test(self, x: np.ndarray, y: np.ndarray,
                       z: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """Fisher Z变换检验"""
        if z is not None:
            x_res = self._residualize(x, z)
            y_res = self._residualize(y, z)
            corr, _ = pearsonr(x_res, y_res)
        else:
            corr, _ = pearsonr(x, y)
        
        n = len(x)
        fisher_z = 0.5 * np.log((1 + corr) / (1 - corr))
        statistic = fisher_z * np.sqrt(n - 3)
        pvalue = 2 * (1 - stats.norm.cdf(abs(statistic)))
        
        return pvalue > self.alpha, pvalue
    
    def _hsic_test(self, x: np.ndarray, y: np.ndarray,
                   z: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """HSIC独立性检验"""
        if z is not None:
            x_res = self._residualize(x, z)
            y_res = self._residualize(y, z)
        else:
            x_res, y_res = x, y
        
        hsic_stat = self._compute_hsic(x_res, y_res)
        # 使用置换检验估计p值
        pvalue = self._hsic_permutation_test(x_res, y_res, hsic_stat)
        return pvalue > self.alpha, pvalue
    
    def _compute_hsic(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算HSIC统计量"""
        n = len(x)
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        # RBF核
        K = self._rbf_kernel(x, x)
        L = self._rbf_kernel(y, y)
        
        H = np.eye(n) - np.ones((n, n)) / n
        Kc = H @ K @ H
        
        hsic = np.trace(Kc @ L) / (n - 1)**2
        return hsic
    
    def _rbf_kernel(self, x: np.ndarray, y: np.ndarray, gamma: float = None) -> np.ndarray:
        """RBF核函数"""
        if gamma is None:
            gamma = 1.0 / x.shape[1]
        
        dist = np.sum(x**2, axis=1).reshape(-1, 1) + \
               np.sum(y**2, axis=1) - 2 * x @ y.T
        return np.exp(-gamma * dist)
    
    def _hsic_permutation_test(self, x: np.ndarray, y: np.ndarray,
                                observed_stat: float, n_permutations: int = 100) -> float:
        """HSIC置换检验"""
        count = 0
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y)
            perm_stat = self._compute_hsic(x, y_perm)
            if perm_stat >= observed_stat:
                count += 1
        return (count + 1) / (n_permutations + 1)
    
    def _residualize(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """对x关于z进行回归并返回残差"""
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        
        # 添加常数项
        Z = np.column_stack([np.ones(len(z)), z])
        
        # 最小二乘回归
        beta = np.linalg.lstsq(Z, x, rcond=None)[0]
        x_pred = Z @ beta
        
        return x - x_pred


class PCAlgorithm:
    """
    PC算法 - Peter-Clark算法
    
    基于条件独立性检验的因果结构学习算法
    """
    
    def __init__(self, independence_tester: IndependenceTester = None,
                 max_cond_vars: int = None,
                 verbose: bool = False):
        self.tester = independence_tester or IndependenceTester()
        self.max_cond_vars = max_cond_vars
        self.verbose = verbose
        self.separation_set: Dict[Tuple[str, str], Set[str]] = {}
        
    def fit(self, data: pd.DataFrame) -> CausalGraph:
        """
        学习因果结构
        
        Args:
            data: 观测数据，列名为变量名
            
        Returns:
            学习到的因果图
        """
        variables = list(data.columns)
        n_vars = len(variables)
        
        # 初始化完全无向图
        adjacency = {v: set(variables) - {v} for v in variables}
        
        # 第一步：骨架学习
        adjacency, self.separation_set = self._skeleton_learning(data, adjacency)
        
        # 第二步：定向边
        graph = self._orient_edges(adjacency, variables)
        
        return graph
    
    def _skeleton_learning(self, data: pd.DataFrame, 
                           adjacency: Dict[str, Set[str]]) -> Tuple[Dict[str, Set[str]], Dict]:
        """骨架学习 - 识别条件独立性"""
        variables = list(adjacency.keys())
        n = len(variables)
        separation_set = {}
        
        depth = 0
        while True:
            if self.max_cond_vars is not None and depth > self.max_cond_vars:
                break
            
            removed = False
            for x in variables:
                for y in list(adjacency[x]):
                    if y not in adjacency[x]:
                        continue
                    
                    # 寻找大小为depth的条件集
                    neighbors = adjacency[x] - {y}
                    if len(neighbors) < depth:
                        continue
                    
                    for cond_set in itertools.combinations(neighbors, depth):
                        cond_set_names = set(cond_set)
                        
                        # 条件独立性检验
                        x_data = data[x].values
                        y_data = data[y].values
                        
                        if depth > 0:
                            z_data = data[list(cond_set)].values
                        else:
                            z_data = None
                        
                        is_independent, pvalue = self.tester.test(x_data, y_data, z_data)
                        
                        if is_independent:
                            if self.verbose:
                                print(f"{x} ⊥⊥ {y} | {cond_set}, p={pvalue:.4f}")
                            
                            adjacency[x].remove(y)
                            adjacency[y].remove(x)
                            separation_set[(x, y)] = cond_set_names
                            separation_set[(y, x)] = cond_set_names
                            removed = True
                            break
                    
                    if removed:
                        break
                if removed:
                    break
            
            if not removed:
                break
            depth += 1
        
        return adjacency, separation_set
    
    def _orient_edges(self, adjacency: Dict[str, Set[str]], 
                      variables: List[str]) -> CausalGraph:
        """定向边 - 应用定向规则"""
        graph = CausalGraph(variables)
        
        # 首先添加所有无向边
        oriented = set()
        for x in variables:
            for y in adjacency[x]:
                if (y, x) not in oriented:
                    edge = CausalEdge(x, y, edge_type="undirected")
                    graph.add_edge(edge)
                    oriented.add((x, y))
        
        # 规则1: 定向v-结构
        self._orient_v_structures(graph, adjacency)
        
        # 规则2-4: 传播定向
        self._propagate_orientations(graph)
        
        return graph
    
    def _orient_v_structures(self, graph: CausalGraph, 
                             adjacency: Dict[str, Set[str]]):
        """定向v-结构 ( collider )"""
        for x in graph.variables:
            for z in graph.variables:
                if z == x:
                    continue
                for y in graph.variables:
                    if y == x or y == z:
                        continue
                    
                    # 检查是否是v-结构: x - z - y, x与y不相邻
                    if (z in adjacency[x] and z in adjacency[y] and 
                        y not in adjacency[x]):
                        
                        # 检查z是否在分离集中
                        sep_set = self.separation_set.get((x, y), set())
                        if z not in sep_set:
                            # 定向为 x -> z <- y
                            self._orient_edge(graph, x, z)
                            self._orient_edge(graph, y, z)
    
    def _orient_edge(self, graph: CausalGraph, source: str, target: str):
        """定向边"""
        # 移除无向边，添加有向边
        edges_to_remove = [e for e in graph.edges 
                          if ((e.source == source and e.target == target) or
                              (e.source == target and e.target == source)) and
                             e.edge_type == "undirected"]
        for e in edges_to_remove:
            graph.edges.remove(e)
        
        new_edge = CausalEdge(source, target, edge_type="directed")
        graph.add_edge(new_edge)
    
    def _propagate_orientations(self, graph: CausalGraph):
        """传播定向规则"""
        changed = True
        while changed:
            changed = False
            
            # 规则2: 如果 a -> b - c 且 a与c不相邻，则定向 b -> c
            for edge in list(graph.edges):
                if edge.edge_type == "undirected":
                    a, b = edge.source, edge.target
                    # 检查是否有入边到a或b
                    for e2 in graph.edges:
                        if e2.edge_type == "directed":
                            if e2.target == a:  # x -> a - b
                                x = e2.source
                                if not graph.graph.has_edge(x, b):
                                    self._orient_edge(graph, a, b)
                                    changed = True
                                    break
            
            # 规则3: 避免有向环
            try:
                cycles = list(nx.simple_cycles(graph.graph))
                for cycle in cycles:
                    # 尝试打破环
                    for i in range(len(cycle)):
                        a, b = cycle[i], cycle[(i+1) % len(cycle)]
                        edge = next((e for e in graph.edges 
                                   if e.source == a and e.target == b), None)
                        if edge and edge.edge_type == "undirected":
                            # 反向定向
                            self._orient_edge(graph, b, a)
                            changed = True
                            break
            except:
                pass


class GESAlgorithm:
    """
    GES算法 - Greedy Equivalence Search
    
    基于评分函数的贪婪搜索算法
    """
    
    def __init__(self, score_type: str = "bic", 
                 max_parents: int = None,
                 verbose: bool = False):
        self.score_type = score_type
        self.max_parents = max_parents
        self.verbose = verbose
        self.cache = {}
        
    def fit(self, data: pd.DataFrame) -> CausalGraph:
        """
        学习因果结构
        
        Args:
            data: 观测数据
            
        Returns:
            学习到的因果图
        """
        variables = list(data.columns)
        n_samples = len(data)
        
        # 初始化空图
        current_graph = self._empty_graph(variables)
        current_score = self._score_graph(current_graph, data)
        
        # 前向阶段：添加边
        improved = True
        while improved:
            improved = False
            best_score = current_score
            best_graph = current_graph
            
            for x in variables:
                for y in variables:
                    if x == y:
                        continue
                    
                    # 尝试添加边 x -> y
                    if not self._would_create_cycle(current_graph, x, y):
                        new_graph = self._copy_graph(current_graph)
                        new_graph.add_edge(CausalEdge(x, y))
                        
                        score = self._score_graph(new_graph, data)
                        
                        if score > best_score:
                            best_score = score
                            best_graph = new_graph
                            improved = True
            
            if improved:
                current_graph = best_graph
                current_score = best_score
                if self.verbose:
                    print(f"Forward: score = {current_score:.2f}")
        
        # 后向阶段：删除边
        improved = True
        while improved:
            improved = False
            best_score = current_score
            
            for edge in list(current_graph.edges):
                new_graph = self._copy_graph(current_graph)
                new_graph.remove_edge(edge.source, edge.target)
                
                score = self._score_graph(new_graph, data)
                
                if score > best_score:
                    best_score = score
                    best_graph = new_graph
                    improved = True
            
            if improved:
                current_graph = best_graph
                current_score = best_score
                if self.verbose:
                    print(f"Backward: score = {current_score:.2f}")
        
        return current_graph
    
    def _empty_graph(self, variables: List[str]) -> CausalGraph:
        """创建空图"""
        return CausalGraph(variables)
    
    def _copy_graph(self, graph: CausalGraph) -> CausalGraph:
        """复制图"""
        new_graph = CausalGraph(graph.variables)
        for edge in graph.edges:
            new_graph.add_edge(CausalEdge(edge.source, edge.target, 
                                         edge.weight, edge.edge_type))
        return new_graph
    
    def _score_graph(self, graph: CausalGraph, data: pd.DataFrame) -> float:
        """计算图的评分"""
        if self.score_type == "bic":
            return self._bic_score(graph, data)
        elif self.score_type == "aic":
            return self._aic_score(graph, data)
        elif self.score_type == "bdeu":
            return self._bdeu_score(graph, data)
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")
    
    def _bic_score(self, graph: CausalGraph, data: pd.DataFrame) -> float:
        """BIC评分 (贝叶斯信息准则)"""
        n = len(data)
        score = 0
        
        for var in graph.variables:
            parents = graph.get_parents(var)
            
            # 计算局部BIC分数
            local_score = self._local_bic(var, parents, data)
            score += local_score
        
        return score
    
    def _local_bic(self, var: str, parents: List[str], data: pd.DataFrame) -> float:
        """局部BIC分数"""
        n = len(data)
        
        if len(parents) == 0:
            # 无父节点时，拟合截距模型
            var_data = data[var].values
            residual_var = np.var(var_data)
            if residual_var < 1e-10:
                residual_var = 1e-10
            log_likelihood = -0.5 * n * (1 + np.log(2 * np.pi * residual_var))
            k = 1  # 1个参数 (方差)
        else:
            # 线性回归
            y = data[var].values
            X = data[parents].values
            
            # 添加常数项
            X = np.column_stack([np.ones(n), X])
            
            # 拟合
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            residual_var = np.var(residuals)
            if residual_var < 1e-10:
                residual_var = 1e-10
            
            log_likelihood = -0.5 * n * (1 + np.log(2 * np.pi * residual_var))
            k = len(parents) + 2  # 回归系数 + 截距 + 方差
        
        bic = log_likelihood - 0.5 * k * np.log(n)
        return bic
    
    def _aic_score(self, graph: CausalGraph, data: pd.DataFrame) -> float:
        """AIC评分"""
        # 类似于BIC但使用2k惩罚
        n = len(data)
        score = 0
        
        for var in graph.variables:
            parents = graph.get_parents(var)
            bic_score = self._local_bic(var, parents, data)
            # 转换AIC
            k = len(parents) + 2 if parents else 1
            aic = bic_score + 0.5 * k * np.log(n) - k
            score += aic
        
        return score
    
    def _bdeu_score(self, graph: CausalGraph, data: pd.DataFrame) -> float:
        """BDeu评分 (贝叶斯狄利克雷等效均匀)"""
        # 简化的BDeu实现
        return self._bic_score(graph, data)
    
    def _would_create_cycle(self, graph: CausalGraph, source: str, target: str) -> bool:
        """检查添加边是否会创建环"""
        # 如果target可以到达source，则添加source->target会创建环
        return source in graph.get_descendants(target)


class NOTEARSAlgorithm:
    """
    NOTEARS算法 - 连续优化方法
    
    使用可微分优化的因果发现方法
    """
    
    def __init__(self, lambda1: float = 0.1,
                 lambda2: float = 0.0,
                 max_iter: int = 100,
                 h_tol: float = 1e-8,
                 rho_max: float = 1e+16,
                 w_threshold: float = 0.3,
                 verbose: bool = False):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.verbose = verbose
        
    def fit(self, data: pd.DataFrame) -> CausalGraph:
        """
        学习因果结构
        
        Args:
            data: 观测数据
            
        Returns:
            学习到的因果图
        """
        variables = list(data.columns)
        X = data.values
        n, d = X.shape
        
        # 标准化数据
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # 使用梯度下降优化
        W = self._notears_linear(X, lambda1=self.lambda1, lambda2=self.lambda2)
        
        # 阈值化
        W[np.abs(W) < self.w_threshold] = 0
        
        # 构建因果图
        graph = CausalGraph(variables)
        graph.from_adjacency_matrix(W, variables)
        
        return graph
    
    def _notears_linear(self, X: np.ndarray, lambda1: float, lambda2: float) -> np.ndarray:
        """NOTEARS线性版本"""
        n, d = X.shape
        
        # 初始化权重矩阵
        W = np.zeros((d, d))
        
        # 增广拉格朗日参数
        rho = 1.0
        alpha = 0.0
        
        # 使用坐标下降
        for iteration in range(self.max_iter):
            # 求解W
            W_new = self._solve_W(X, W, rho, alpha, lambda1, lambda2)
            
            # 计算无环约束
            h = self._compute_h(W_new)
            
            if self.verbose and iteration % 10 == 0:
                loss = self._loss(X, W_new)
                print(f"Iter {iteration}: loss={loss:.4f}, h={h:.6f}, rho={rho:.2e}")
            
            # 检查收敛
            if h < self.h_tol and iteration > 0:
                break
            
            # 更新对偶变量
            alpha += rho * h
            
            # 增加惩罚参数
            if h > 0.25 * self._compute_h(W):
                rho *= 10
            
            if rho > self.rho_max:
                break
            
            W = W_new
        
        return W
    
    def _solve_W(self, X: np.ndarray, W: np.ndarray, rho: float,
                 alpha: float, lambda1: float, lambda2: float) -> np.ndarray:
        """使用近端梯度法求解W"""
        n, d = X.shape
        
        # 最小二乘损失 + L1正则 + 无环约束
        # 使用简化版本：迭代更新
        W_new = W.copy()
        
        for _ in range(100):  # 内循环
            # 计算梯度
            grad = self._gradient(X, W_new, rho, alpha)
            
            # 梯度下降步骤
            step_size = 0.001
            W_temp = W_new - step_size * grad
            
            # 近端算子 (软阈值) - L1正则
            W_temp = self._soft_threshold(W_temp, step_size * lambda1)
            
            # 确保对角线为0 (无自环)
            np.fill_diagonal(W_temp, 0)
            
            W_new = W_temp
        
        return W_new
    
    def _gradient(self, X: np.ndarray, W: np.ndarray, rho: float, alpha: float) -> np.ndarray:
        """计算梯度"""
        n, d = X.shape
        
        # 最小二乘梯度
        grad_ls = (2.0 / n) * (X.T @ X @ W - X.T @ X)
        
        # 无环约束梯度
        grad_h = rho * self._h_gradient(W) + alpha * self._h_gradient(W)
        
        return grad_ls + grad_h
    
    def _loss(self, X: np.ndarray, W: np.ndarray) -> float:
        """计算损失函数"""
        n, d = X.shape
        loss = (1.0 / (2 * n)) * np.sum((X @ W - X) ** 2)
        return loss
    
    def _compute_h(self, W: np.ndarray) -> float:
        """计算无环约束 h(W) = tr(e^(W∘W)) - d"""
        d = W.shape[0]
        # 简化的迹指数计算
        M = W * W
        # 使用矩阵指数的近似
        h = np.trace(np.eye(d) + M + 0.5 * M @ M) - d
        return h
    
    def _h_gradient(self, W: np.ndarray) -> np.ndarray:
        """计算h的梯度"""
        d = W.shape[0]
        M = W * W
        # 近似梯度
        grad = 2 * W * (np.eye(d) + M)
        return grad
    
    def _soft_threshold(self, W: np.ndarray, threshold: float) -> np.ndarray:
        """软阈值算子"""
        return np.sign(W) * np.maximum(np.abs(W) - threshold, 0)


class InterventionEffectEstimator:
    """
    干预效应估计器
    
    估计do-演算中的干预效应
    """
    
    def __init__(self, causal_graph: CausalGraph = None):
        self.graph = causal_graph
        self.models: Dict[str, Callable] = {}
        
    def fit(self, data: pd.DataFrame, graph: CausalGraph = None):
        """
        拟合结构方程模型
        
        Args:
            data: 观测数据
            graph: 因果图（如果初始化时未提供）
        """
        if graph is not None:
            self.graph = graph
        
        if self.graph is None:
            raise ValueError("Causal graph must be provided")
        
        # 为每个变量拟合结构方程
        for var in self.graph.variables:
            parents = self.graph.get_parents(var)
            
            if len(parents) == 0:
                # 根节点：估计边际分布
                self.models[var] = self._fit_marginal(data[var].values)
            else:
                # 拟合条件分布 P(var | parents)
                self.models[var] = self._fit_conditional(
                    data[var].values, 
                    data[parents].values,
                    parents
                )
    
    def _fit_marginal(self, y: np.ndarray) -> Callable:
        """拟合边际分布"""
        mean = np.mean(y)
        std = np.std(y)
        
        def sample(n_samples: int = 1) -> np.ndarray:
            return np.random.normal(mean, std, n_samples)
        
        return sample
    
    def _fit_conditional(self, y: np.ndarray, X: np.ndarray,
                        parent_names: List[str]) -> Callable:
        """拟合条件分布 P(y | X)"""
        # 线性回归模型
        n = len(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_design = np.column_stack([np.ones(n), X])
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
        
        # 残差标准差
        y_pred = X_design @ beta
        residuals = y - y_pred
        sigma = np.std(residuals)
        
        def conditional_sample(parent_values: Dict[str, float], 
                              n_samples: int = 1) -> np.ndarray:
            """给定父节点值，采样子节点"""
            x = [1.0]  # 截距
            for p in parent_names:
                x.append(parent_values.get(p, 0))
            x = np.array(x)
            
            mean = x @ beta
            return np.random.normal(mean, sigma, n_samples)
        
        return conditional_sample
    
    def estimate_ate(self, treatment: str, outcome: str,
                     intervention_values: List[float] = None) -> Dict[float, float]:
        """
        估计平均处理效应 (ATE)
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            intervention_values: 干预值列表
            
        Returns:
            每个干预值对应的期望结果
        """
        if intervention_values is None:
            intervention_values = [0, 1]
        
        results = {}
        
        for value in intervention_values:
            # 模拟干预 do(treatment=value)
            expected_outcome = self._simulate_intervention(
                Intervention(treatment, value),
                outcome
            )
            results[value] = expected_outcome
        
        return results
    
    def _simulate_intervention(self, intervention: Intervention,
                               target_var: str,
                               n_samples: int = 1000) -> float:
        """模拟干预"""
        results = []
        
        for _ in range(n_samples):
            # 从修改后的模型生成样本
            sample = self._generate_sample_with_intervention(intervention)
            results.append(sample.get(target_var, 0))
        
        return np.mean(results)
    
    def _generate_sample_with_intervention(self, 
                                           intervention: Intervention) -> Dict[str, float]:
        """在干预下生成样本"""
        sample = {}
        
        # 拓扑排序
        sorted_vars = list(nx.topological_sort(self.graph.graph))
        
        for var in sorted_vars:
            if var == intervention.variable:
                # 应用干预
                sample[var] = intervention.value
            else:
                parents = self.graph.get_parents(var)
                parent_values = {p: sample.get(p, 0) for p in parents}
                
                if len(parents) == 0:
                    # 根节点
                    sample[var] = self.models[var](1)[0]
                else:
                    # 条件采样
                    sample[var] = self.models[var](parent_values, 1)[0]
        
        return sample
    
    def estimate_cate(self, treatment: str, outcome: str,
                      covariates: List[str],
                      data: pd.DataFrame) -> pd.DataFrame:
        """
        估计条件平均处理效应 (CATE)
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            covariates: 协变量
            data: 数据
            
        Returns:
            每个样本的CATE估计
        """
        cates = []
        
        for idx, row in data.iterrows():
            # 固定协变量，改变处理
            base_condition = {c: row[c] for c in covariates}
            
            # E[Y | do(T=1), X=x]
            y1 = self._simulate_intervention_with_context(
                Intervention(treatment, 1), outcome, base_condition
            )
            
            # E[Y | do(T=0), X=x]
            y0 = self._simulate_intervention_with_context(
                Intervention(treatment, 0), outcome, base_condition
            )
            
            cate = y1 - y0
            cates.append(cate)
        
        result = data.copy()
        result['CATE'] = cates
        return result
    
    def _simulate_intervention_with_context(self, intervention: Intervention,
                                           target_var: str,
                                           context: Dict[str, float],
                                           n_samples: int = 100) -> float:
        """在特定上下文下模拟干预"""
        results = []
        
        for _ in range(n_samples):
            sample = self._generate_sample_with_context(intervention, context)
            results.append(sample.get(target_var, 0))
        
        return np.mean(results)
    
    def _generate_sample_with_context(self, intervention: Intervention,
                                      context: Dict[str, float]) -> Dict[str, float]:
        """在特定上下文中生成样本"""
        sample = context.copy()
        
        # 获取未在上下文中的变量
        remaining_vars = [v for v in self.graph.variables 
                         if v not in context and v != intervention.variable]
        
        # 设置干预变量
        sample[intervention.variable] = intervention.value
        
        # 按拓扑顺序生成其余变量
        for var in remaining_vars:
            parents = self.graph.get_parents(var)
            parent_values = {p: sample.get(p, 0) for p in parents}
            
            if len(parents) == 0:
                sample[var] = self.models[var](1)[0]
            else:
                sample[var] = self.models[var](parent_values, 1)[0]
        
        return sample


class CounterfactualInference:
    """
    反事实推理
    
    基于结构因果模型进行反事实推理
    """
    
    def __init__(self, causal_graph: CausalGraph):
        self.graph = causal_graph
        self.structural_equations: Dict[str, Callable] = {}
        self.noise_terms: Dict[str, np.ndarray] = {}
        
    def fit(self, data: pd.DataFrame):
        """
        拟合结构方程模型
        
        Args:
            data: 观测数据
        """
        n_samples = len(data)
        
        for var in self.graph.variables:
            parents = self.graph.get_parents(var)
            
            if len(parents) == 0:
                # 外生变量
                self.structural_equations[var] = lambda noise: noise
                self.noise_terms[var] = data[var].values
            else:
                # 内生变量：拟合结构方程并提取噪声
                y = data[var].values
                X = data[parents].values
                
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                
                X_design = np.column_stack([np.ones(n_samples), X])
                beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
                
                # 预测和噪声
                y_pred = X_design @ beta
                noise = y - y_pred
                
                self.noise_terms[var] = noise
                
                # 创建结构方程函数
                def create_equation(beta_coef, parent_names):
                    def equation(parent_values, noise):
                        result = beta_coef[0]  # 截距
                        for i, p in enumerate(parent_names):
                            result += beta_coef[i+1] * parent_values.get(p, 0)
                        return result + noise
                    return equation
                
                self.structural_equations[var] = create_equation(beta, parents)
    
    def infer(self, query: CounterfactualQuery) -> float:
        """
        执行反事实推理
        
        Args:
            query: 反事实查询
            
        Returns:
            反事实结果
        """
        # 步骤1：推断 (Abduction) - 从事实推断外生变量
        inferred_noise = self._infer_noise(query.factual_evidence)
        
        # 步骤2：行动 (Action) - 修改模型以反映干预
        modified_graph = self._apply_intervention(query.hypothetical_intervention)
        
        # 步骤3：预测 (Prediction) - 在新模型中计算结果
        counterfactual_result = self._predict(modified_graph, 
                                              inferred_noise,
                                              query.hypothetical_intervention,
                                              query.target_variable)
        
        return counterfactual_result
    
    def _infer_noise(self, evidence: Dict[str, float]) -> Dict[str, float]:
        """从证据推断噪声项"""
        # 对于每个观测变量，找到最接近的样本索引
        inferred = {}
        
        # 计算与证据最接近的样本
        distances = np.zeros(len(self.noise_terms[list(self.noise_terms.keys())[0]]))
        for var, value in evidence.items():
            if var in self.noise_terms:
                # 找到最近的噪声值
                distances += (self.noise_terms[var] - value) ** 2
        
        closest_idx = np.argmin(distances)
        
        # 提取该样本的所有噪声项
        for var in self.graph.variables:
            inferred[var] = self.noise_terms[var][closest_idx]
        
        return inferred
    
    def _apply_intervention(self, intervention: Intervention) -> CausalGraph:
        """应用干预到因果图"""
        # 创建图的副本
        modified = CausalGraph(self.graph.variables)
        
        for edge in self.graph.edges:
            # 移除指向干预变量的入边
            if edge.target == intervention.variable:
                continue
            modified.add_edge(edge)
        
        return modified
    
    def _predict(self, graph: CausalGraph,
                noise: Dict[str, float],
                intervention: Intervention,
                target: str) -> float:
        """在修改后的模型中预测"""
        values = {}
        
        # 按拓扑顺序计算
        sorted_vars = list(nx.topological_sort(graph.graph))
        
        for var in sorted_vars:
            if var == intervention.variable:
                values[var] = intervention.value
            else:
                parents = graph.get_parents(var)
                parent_values = {p: values.get(p, 0) for p in parents}
                
                if len(parents) == 0:
                    # 外生变量
                    values[var] = noise.get(var, 0)
                else:
                    # 应用结构方程
                    eq = self.structural_equations[var]
                    values[var] = eq(parent_values, noise.get(var, 0))
        
        return values.get(target, 0)
    
    def batch_infer(self, queries: List[CounterfactualQuery]) -> List[float]:
        """批量反事实推理"""
        return [self.infer(q) for q in queries]


class CausalDiscoveryPipeline:
    """
    因果发现管道
    
    整合多个因果发现算法的完整管道
    """
    
    def __init__(self, algorithm: CausalAlgorithm = CausalAlgorithm.PC,
                 independence_test: IndependenceTest = IndependenceTest.PEARSON,
                 alpha: float = 0.05,
                 verbose: bool = False):
        self.algorithm_type = algorithm
        self.test_type = independence_test
        self.alpha = alpha
        self.verbose = verbose
        self.graph: Optional[CausalGraph] = None
        self.intervention_estimator: Optional[InterventionEffectEstimator] = None
        self.counterfactual_engine: Optional[CounterfactualInference] = None
        
    def fit(self, data: pd.DataFrame) -> CausalGraph:
        """
        执行因果发现
        
        Args:
            data: 观测数据
            
        Returns:
            学习到的因果图
        """
        if self.verbose:
            print(f"Running {self.algorithm_type.value} algorithm...")
        
        if self.algorithm_type == CausalAlgorithm.PC:
            tester = IndependenceTester(self.test_type, self.alpha)
            algo = PCAlgorithm(tester, verbose=self.verbose)
        elif self.algorithm_type == CausalAlgorithm.GES:
            algo = GESAlgorithm(verbose=self.verbose)
        elif self.algorithm_type == CausalAlgorithm.NOTEARS:
            algo = NOTEARSAlgorithm(verbose=self.verbose)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_type}")
        
        self.graph = algo.fit(data)
        
        if self.verbose:
            print(f"Discovered {len(self.graph.edges)} edges")
        
        return self.graph
    
    def estimate_intervention_effect(self, data: pd.DataFrame) -> InterventionEffectEstimator:
        """估计干预效应"""
        if self.graph is None:
            raise ValueError("Must run fit() first")
        
        self.intervention_estimator = InterventionEffectEstimator(self.graph)
        self.intervention_estimator.fit(data)
        
        return self.intervention_estimator
    
    def setup_counterfactual(self, data: pd.DataFrame) -> CounterfactualInference:
        """设置反事实推理"""
        if self.graph is None:
            raise ValueError("Must run fit() first")
        
        self.counterfactual_engine = CounterfactualInference(self.graph)
        self.counterfactual_engine.fit(data)
        
        return self.counterfactual_engine
    
    def analyze_causal_effects(self, treatment: str, outcome: str,
                               data: pd.DataFrame) -> Dict:
        """
        全面分析因果效应
        
        Returns:
            包含各种因果效应度量的字典
        """
        results = {
            'treatment': treatment,
            'outcome': outcome,
        }
        
        # ATE估计
        if self.intervention_estimator is None:
            self.estimate_intervention_effect(data)
        
        ate_results = self.intervention_estimator.estimate_ate(treatment, outcome)
        results['ATE'] = ate_results.get(1, 0) - ate_results.get(0, 0)
        
        # 因果路径分析
        paths = self._find_causal_paths(treatment, outcome)
        results['causal_paths'] = paths
        
        # 中介分析
        mediators = self._find_mediators(treatment, outcome)
        results['mediators'] = mediators
        
        return results
    
    def _find_causal_paths(self, source: str, target: str) -> List[List[str]]:
        """找到所有因果路径"""
        try:
            paths = list(nx.all_simple_paths(self.graph.graph, source, target))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def _find_mediators(self, treatment: str, outcome: str) -> List[str]:
        """找到潜在的中介变量"""
        # 同时是treatment的后代和outcome的祖先的变量
        treatment_descendants = self.graph.get_descendants(treatment)
        outcome_ancestors = self.graph.get_ancestors(outcome)
        
        mediators = treatment_descendants & outcome_ancestors
        return list(mediators)
    
    def validate_with_instrumental_variable(self, treatment: str, outcome: str,
                                           instrument: str, data: pd.DataFrame) -> Dict:
        """
        使用工具变量验证因果效应
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            instrument: 工具变量
            data: 数据
            
        Returns:
            包含2SLS估计结果的字典
        """
        from sklearn.linear_model import LinearRegression
        
        # 第一阶段：T ~ Z
        Z = data[[instrument]].values
        T = data[treatment].values
        
        stage1 = LinearRegression()
        stage1.fit(Z, T)
        T_pred = stage1.predict(Z)
        
        # 第二阶段：Y ~ T_pred
        Y = data[outcome].values
        stage2 = LinearRegression()
        stage2.fit(T_pred.reshape(-1, 1), Y)
        
        causal_effect = stage2.coef_[0]
        
        return {
            'treatment': treatment,
            'outcome': outcome,
            'instrument': instrument,
            'causal_effect': causal_effect,
            'stage1_r2': stage1.score(Z, T),
            'stage2_r2': stage2.score(T_pred.reshape(-1, 1), Y)
        }


def example_usage():
    """使用示例"""
    # 生成合成数据
    np.random.seed(42)
    n = 1000
    
    # 模拟因果结构：X -> Z -> Y, X -> Y
    X = np.random.normal(0, 1, n)
    Z = 0.5 * X + np.random.normal(0, 0.5, n)
    Y = 0.3 * X + 0.7 * Z + np.random.normal(0, 0.3, n)
    
    data = pd.DataFrame({
        'X': X,
        'Z': Z,
        'Y': Y
    })
    
    print("=" * 60)
    print("因果发现示例")
    print("=" * 60)
    print(f"\n真实因果结构: X -> Z -> Y, X -> Y")
    print(f"样本量: {n}\n")
    
    # PC算法
    print("-" * 40)
    print("PC算法")
    print("-" * 40)
    
    pipeline = CausalDiscoveryPipeline(
        algorithm=CausalAlgorithm.PC,
        verbose=True
    )
    graph = pipeline.fit(data)
    
    print(f"\n发现的边:")
    for edge in graph.edges:
        print(f"  {edge.source} {'-->' if edge.edge_type == 'directed' else '---'} {edge.target}")
    
    # GES算法
    print("\n" + "-" * 40)
    print("GES算法")
    print("-" * 40)
    
    pipeline_ges = CausalDiscoveryPipeline(
        algorithm=CausalAlgorithm.GES,
        verbose=True
    )
    graph_ges = pipeline_ges.fit(data)
    
    print(f"\n发现的边:")
    for edge in graph_ges.edges:
        print(f"  {edge.source} {'-->' if edge.edge_type == 'directed' else '---'} {edge.target}")
    
    # 干预效应估计
    print("\n" + "-" * 40)
    print("干预效应估计")
    print("-" * 40)
    
    pipeline.estimate_intervention_effect(data)
    ate_results = pipeline.intervention_estimator.estimate_ate('X', 'Y')
    print(f"\nATE of X on Y:")
    print(f"  E[Y | do(X=0)] = {ate_results[0]:.4f}")
    print(f"  E[Y | do(X=1)] = {ate_results[1]:.4f}")
    print(f"  ATE = {ate_results[1] - ate_results[0]:.4f}")
    
    # 中介分析
    print("\n" + "-" * 40)
    print("因果效应分析")
    print("-" * 40)
    
    analysis = pipeline.analyze_causal_effects('X', 'Y', data)
    print(f"\n中介变量: {analysis['mediators']}")
    print(f"因果路径: {analysis['causal_paths']}")
    
    # 反事实推理
    print("\n" + "-" * 40)
    print("反事实推理")
    print("-" * 40)
    
    pipeline.setup_counterfactual(data)
    
    # 反事实查询：如果X=2，Y会是多少？
    query = CounterfactualQuery(
        factual_evidence={'X': 0, 'Y': data['Y'].mean()},
        hypothetical_intervention=Intervention('X', 2.0),
        target_variable='Y'
    )
    
    result = pipeline.counterfactual_engine.infer(query)
    print(f"\n反事实查询: 如果X=2，Y会是多少？")
    print(f"反事实结果: Y = {result:.4f}")
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
