"""
贝叶斯网络 - Bayesian Network
实现贝叶斯网络的构建、学习和推理
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import itertools
import warnings


class NodeType(Enum):
    """节点类型"""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


@dataclass
class BNNode:
    """贝叶斯网络节点"""
    name: str
    node_type: NodeType = NodeType.DISCRETE
    parents: List[str] = field(default_factory=list)
    states: Optional[List[str]] = None  # 离散变量的状态
    
    # 条件概率表 (CPT) 或条件概率分布 (CPD)
    cpt: Optional[np.ndarray] = None
    cpd_mean: Optional[Callable] = None  # 连续变量的均值函数
    cpd_std: float = 1.0  # 连续变量的标准差
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if not isinstance(other, BNNode):
            return False
        return self.name == other.name


class BayesianNetwork:
    """
    贝叶斯网络
    支持离散和连续变量的概率图模型
    """
    
    def __init__(self):
        self.nodes: Dict[str, BNNode] = {}
        self.edges: List[Tuple[str, str]] = []
        self.node_order: List[str] = []  # 拓扑排序
    
    def add_node(
        self,
        name: str,
        node_type: NodeType = NodeType.DISCRETE,
        states: Optional[List[str]] = None
    ) -> BNNode:
        """添加节点"""
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists")
        
        node = BNNode(name=name, node_type=node_type, states=states)
        self.nodes[name] = node
        return node
    
    def add_edge(self, parent: str, child: str):
        """添加边"""
        if parent not in self.nodes:
            raise ValueError(f"Parent node {parent} not found")
        if child not in self.nodes:
            raise ValueError(f"Child node {child} not found")
        
        self.nodes[child].parents.append(parent)
        self.edges.append((parent, child))
    
    def set_cpt(self, node_name: str, cpt: np.ndarray):
        """设置条件概率表"""
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found")
        
        node = self.nodes[node_name]
        n_parents = len(node.parents)
        
        if n_parents == 0:
            # 边缘概率
            if node.states:
                expected_shape = (len(node.states),)
            else:
                expected_shape = cpt.shape
        else:
            # 条件概率
            parent_states = [len(self.nodes[p].states) for p in node.parents]
            if node.states:
                expected_shape = tuple(parent_states + [len(node.states)])
            else:
                expected_shape = tuple(parent_states + [cpt.shape[-1]])
        
        if cpt.shape != expected_shape:
            warnings.warn(f"CPT shape {cpt.shape} doesn't match expected {expected_shape}")
        
        node.cpt = cpt
    
    def get_parents(self, node_name: str) -> List[str]:
        """获取父节点"""
        return self.nodes[node_name].parents
    
    def get_children(self, node_name: str) -> List[str]:
        """获取子节点"""
        children = []
        for parent, child in self.edges:
            if parent == node_name:
                children.append(child)
        return children
    
    def get_ancestors(self, node_name: str) -> Set[str]:
        """获取祖先节点"""
        ancestors = set()
        stack = [node_name]
        
        while stack:
            current = stack.pop()
            parents = self.get_parents(current)
            for p in parents:
                if p not in ancestors:
                    ancestors.add(p)
                    stack.append(p)
        
        return ancestors
    
    def get_descendants(self, node_name: str) -> Set[str]:
        """获取后代节点"""
        descendants = set()
        stack = [node_name]
        
        while stack:
            current = stack.pop()
            children = self.get_children(current)
            for c in children:
                if c not in descendants:
                    descendants.add(c)
                    stack.append(c)
        
        return descendants
    
    def is_dag(self) -> bool:
        """检查是否为有向无环图"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for child in self.get_children(node):
                if child not in visited:
                    if has_cycle(child):
                        return True
                elif child in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if has_cycle(node):
                    return False
        
        return True
    
    def topological_sort(self) -> List[str]:
        """拓扑排序"""
        if not self.is_dag():
            raise ValueError("Graph contains cycles")
        
        in_degree = {name: 0 for name in self.nodes}
        for parent, child in self.edges:
            in_degree[child] += 1
        
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        self.node_order = result
        return result
    
    def query(
        self,
        variables: List[str],
        evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        概率查询 (变量消除)
        
        Args:
            variables: 要查询的变量
            evidence: 证据
        
        Returns:
            后验概率分布
        """
        if evidence is None:
            evidence = {}
        
        # 确保拓扑排序
        if not self.node_order:
            self.topological_sort()
        
        # 变量消除（简化实现）
        results = {}
        
        for var in variables:
            if var in evidence:
                # 如果变量在证据中，返回确定的值
                node = self.nodes[var]
                if node.states:
                    dist = np.zeros(len(node.states))
                    state_idx = node.states.index(evidence[var])
                    dist[state_idx] = 1.0
                    results[var] = dist
                else:
                    results[var] = np.array([evidence[var]])
            else:
                # 计算后验分布
                dist = self._infer_posterior(var, evidence)
                results[var] = dist
        
        return results
    
    def _infer_posterior(
        self,
        variable: str,
        evidence: Dict[str, Any]
    ) -> np.ndarray:
        """推断后验分布"""
        node = self.nodes[variable]
        
        if not node.parents:
            # 边缘概率
            if node.cpt is not None:
                return node.cpt
            else:
                return np.ones(len(node.states)) / len(node.states) if node.states else np.array([0.5])
        
        # 有父节点的情况
        parent_values = [evidence.get(p) for p in node.parents]
        
        if all(v is not None for v in parent_values):
            # 所有父节点都有值
            if node.cpt is not None:
                # 查找对应的条件概率
                indices = []
                for i, p in enumerate(node.parents):
                    parent_node = self.nodes[p]
                    if parent_node.states:
                        idx = parent_node.states.index(parent_values[i])
                        indices.append(idx)
                
                cpt_slice = node.cpt
                for idx in indices:
                    cpt_slice = cpt_slice[idx]
                
                return cpt_slice
        
        # 简化处理：返回均匀分布
        if node.states:
            return np.ones(len(node.states)) / len(node.states)
        return np.array([0.5])
    
    def sample(self, n_samples: int = 100) -> np.ndarray:
        """
        从网络中采样
        
        Args:
            n_samples: 样本数
        
        Returns:
            样本数组
        """
        if not self.node_order:
            self.topological_sort()
        
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            
            for node_name in self.node_order:
                node = self.nodes[node_name]
                
                if node.node_type == NodeType.DISCRETE:
                    # 获取条件概率
                    prob = self._get_conditional_prob(node, sample)
                    # 采样
                    if node.states:
                        state_idx = np.random.choice(len(node.states), p=prob)
                        sample[node_name] = node.states[state_idx]
                    else:
                        sample[node_name] = np.random.choice(len(prob), p=prob)
                else:
                    # 连续变量
                    if node.cpd_mean:
                        parent_values = [sample.get(p, 0) for p in node.parents]
                        mean = node.cpd_mean(*parent_values)
                    else:
                        mean = 0
                    sample[node_name] = np.random.normal(mean, node.cpd_std)
            
            samples.append(sample)
        
        return samples
    
    def _get_conditional_prob(
        self,
        node: BNNode,
        sample: Dict[str, Any]
    ) -> np.ndarray:
        """获取节点的条件概率"""
        if not node.parents:
            if node.cpt is not None:
                return node.cpt
            else:
                return np.ones(len(node.states)) / len(node.states) if node.states else np.array([0.5])
        
        parent_values = [sample.get(p) for p in node.parents]
        
        if node.cpt is not None:
            # 构建索引
            indices = []
            for i, p in enumerate(node.parents):
                parent_node = self.nodes[p]
                if parent_node.states and parent_values[i] in parent_node.states:
                    idx = parent_node.states.index(parent_values[i])
                    indices.append(idx)
                else:
                    # 父节点值未知，使用边缘化
                    return np.ones(len(node.states)) / len(node.states) if node.states else np.array([0.5])
            
            cpt_slice = node.cpt
            for idx in indices:
                cpt_slice = cpt_slice[idx]
            
            return cpt_slice
        
        return np.ones(len(node.states)) / len(node.states) if node.states else np.array([0.5])
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "nodes": [
                {
                    "name": n.name,
                    "type": n.node_type.value,
                    "parents": n.parents,
                    "states": n.states
                }
                for n in self.nodes.values()
            ],
            "edges": self.edges
        }


class BNLearner:
    """
    贝叶斯网络学习器
    从数据中学习网络结构和参数
    """
    
    def __init__(self):
        self.data: Optional[np.ndarray] = None
        self.var_names: Optional[List[str]] = None
        self.var_types: Optional[List[NodeType]] = None
    
    def fit_structure(
        self,
        data: np.ndarray,
        var_names: List[str],
        method: str = "hill_climbing",
        max_parents: int = 3
    ) -> BayesianNetwork:
        """
        学习网络结构
        
        Args:
            data: 数据
            var_names: 变量名
            method: 结构学习方法
            max_parents: 最大父节点数
        
        Returns:
            学习到的贝叶斯网络
        """
        self.data = data
        self.var_names = var_names
        
        bn = BayesianNetwork()
        
        # 添加节点
        for name in var_names:
            bn.add_node(name, NodeType.DISCRETE)
        
        if method == "hill_climbing":
            self._hill_climbing(bn, max_parents)
        elif method == "k2":
            self._k2_algorithm(bn, max_parents)
        
        return bn
    
    def _hill_climbing(self, bn: BayesianNetwork, max_parents: int):
        """爬山算法学习结构"""
        n_vars = len(self.var_names)
        best_score = self._score_network(bn)
        
        improved = True
        while improved:
            improved = False
            
            # 尝试所有可能的边添加
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        continue
                    
                    parent = self.var_names[i]
                    child = self.var_names[j]
                    
                    # 检查是否已存在
                    if parent in bn.get_parents(child):
                        continue
                    
                    # 检查父节点数限制
                    if len(bn.get_parents(child)) >= max_parents:
                        continue
                    
                    # 尝试添加边
                    bn.add_edge(parent, child)
                    
                    # 检查是否为DAG
                    if bn.is_dag():
                        score = self._score_network(bn)
                        
                        if score > best_score:
                            best_score = score
                            improved = True
                        else:
                            # 撤销添加
                            bn.nodes[child].parents.remove(parent)
                            bn.edges.remove((parent, child))
                    else:
                        # 撤销添加
                        bn.nodes[child].parents.remove(parent)
                        bn.edges.remove((parent, child))
    
    def _k2_algorithm(self, bn: BayesianNetwork, max_parents: int):
        """K2算法学习结构"""
        # 简化实现
        n_vars = len(self.var_names)
        
        for i, child in enumerate(self.var_names):
            parents = []
            score = self._local_score(child, parents)
            
            while len(parents) < max_parents:
                best_parent = None
                best_score = score
                
                for j in range(i):
                    candidate = self.var_names[j]
                    if candidate in parents:
                        continue
                    
                    new_parents = parents + [candidate]
                    new_score = self._local_score(child, new_parents)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_parent = candidate
                
                if best_parent is not None:
                    parents.append(best_parent)
                    score = best_score
                    bn.add_edge(best_parent, child)
                else:
                    break
    
    def fit_parameters(self, bn: BayesianNetwork, data: np.ndarray):
        """
        学习网络参数（CPT）
        
        Args:
            bn: 贝叶斯网络
            data: 数据
        """
        var_idx = {name: i for i, name in enumerate(self.var_names)}
        
        for node_name, node in bn.nodes.items():
            if node.node_type == NodeType.DISCRETE:
                # 离散变量：学习CPT
                cpt = self._estimate_cpt(node, data, var_idx)
                bn.set_cpt(node_name, cpt)
    
    def _estimate_cpt(
        self,
        node: BNNode,
        data: np.ndarray,
        var_idx: Dict[str, int]
    ) -> np.ndarray:
        """估计条件概率表"""
        node_idx = var_idx[node.name]
        
        if not node.parents:
            # 边缘概率
            if node.states:
                counts = np.zeros(len(node.states))
                for state_idx, state in enumerate(node.states):
                    counts[state_idx] = np.sum(data[:, node_idx] == state)
                # 拉普拉斯平滑
                counts += 1
                return counts / counts.sum()
            else:
                return np.array([0.5, 0.5])
        
        # 条件概率
        parent_states = [len(self.nodes[p].states) for p in node.parents]
        if node.states:
            cpt_shape = tuple(parent_states + [len(node.states)])
        else:
            cpt_shape = tuple(parent_states + [2])
        
        counts = np.zeros(cpt_shape)
        
        # 统计频率
        for row in data:
            parent_indices = []
            for p in node.parents:
                p_idx = var_idx[p]
                p_state = row[p_idx]
                p_node = self.nodes[p]
                if p_node.states:
                    parent_indices.append(p_node.states.index(p_state))
            
            node_state = row[node_idx]
            if node.states:
                node_state_idx = node.states.index(node_state)
            else:
                node_state_idx = int(node_state)
            
            indices = tuple(parent_indices + [node_state_idx])
            counts[indices] += 1
        
        # 拉普拉斯平滑
        counts += 1
        
        # 归一化
        axis = len(counts.shape) - 1
        sums = counts.sum(axis=axis, keepdims=True)
        cpt = counts / (sums + 1e-10)
        
        return cpt
    
    def _score_network(self, bn: BayesianNetwork) -> float:
        """评分网络结构（BIC评分）"""
        score = 0
        
        for node_name in bn.nodes:
            parents = bn.get_parents(node_name)
            score += self._local_score(node_name, parents)
        
        return score
    
    def _local_score(self, node: str, parents: List[str]) -> float:
        """局部评分（BIC）"""
        if self.data is None:
            return 0
        
        node_idx = self.var_names.index(node)
        parent_indices = [self.var_names.index(p) for p in parents]
        
        # 简化的BIC评分
        n = len(self.data)
        
        # 似然部分
        log_likelihood = 0
        
        if not parents:
            # 边缘分布
            unique, counts = np.unique(self.data[:, node_idx], return_counts=True)
            probs = counts / n
            log_likelihood = np.sum(counts * np.log(probs + 1e-10))
        else:
            # 条件分布
            parent_configs = defaultdict(list)
            for i, row in enumerate(self.data):
                config = tuple(row[parent_indices])
                parent_configs[config].append(i)
            
            for config, indices in parent_configs.items():
                subset = self.data[indices, node_idx]
                unique, counts = np.unique(subset, return_counts=True)
                probs = counts / len(indices)
                log_likelihood += np.sum(counts * np.log(probs + 1e-10))
        
        # 惩罚项
        k = len(parents)  # 参数数量简化
        penalty = 0.5 * k * np.log(n)
        
        return log_likelihood - penalty


def demo():
    """演示贝叶斯网络"""
    print("=" * 60)
    print("贝叶斯网络演示")
    print("=" * 60)
    
    # 创建贝叶斯网络
    bn = BayesianNetwork()
    
    # 定义节点
    bn.add_node("BatteryType", NodeType.DISCRETE, ["Li-ion", "Li-poly", "Solid"])
    bn.add_node("Temperature", NodeType.DISCRETE, ["Low", "Medium", "High"])
    bn.add_node("Capacity", NodeType.DISCRETE, ["Low", "Medium", "High"])
    bn.add_node("CycleLife", NodeType.DISCRETE, ["Short", "Medium", "Long"])
    bn.add_node("Safety", NodeType.DISCRETE, ["Poor", "Good", "Excellent"])
    
    # 添加边
    bn.add_edge("BatteryType", "Capacity")
    bn.add_edge("BatteryType", "Safety")
    bn.add_edge("Temperature", "Capacity")
    bn.add_edge("Temperature", "CycleLife")
    bn.add_edge("Capacity", "CycleLife")
    bn.add_edge("Safety", "CycleLife")
    
    print("\n1. 网络结构:")
    print(f"   节点数: {len(bn.nodes)}")
    print(f"   边数: {len(bn.edges)}")
    print(f"   是DAG: {bn.is_dag()}")
    
    # 设置CPT
    bn.set_cpt("BatteryType", np.array([0.5, 0.3, 0.2]))
    bn.set_cpt("Temperature", np.array([0.2, 0.5, 0.3]))
    
    # Capacity 的 CPT
    bn.set_cpt("Capacity", np.array([
        # BatteryType=Li-ion, Temperature
        [[0.1, 0.4, 0.5],   # Low Temp
         [0.1, 0.3, 0.6],   # Medium Temp
         [0.2, 0.4, 0.4]],  # High Temp
        # BatteryType=Li-poly
        [[0.2, 0.5, 0.3],
         [0.15, 0.45, 0.4],
         [0.25, 0.45, 0.3]],
        # BatteryType=Solid
        [[0.3, 0.5, 0.2],
         [0.25, 0.5, 0.25],
         [0.3, 0.45, 0.25]]
    ]))
    
    # Safety 的 CPT
    bn.set_cpt("Safety", np.array([
        [0.1, 0.4, 0.5],  # Li-ion
        [0.2, 0.4, 0.4],  # Li-poly
        [0.05, 0.25, 0.7]  # Solid
    ]))
    
    print("\n2. 概率查询:")
    
    # 查询边缘概率
    result = bn.query(["BatteryType"])
    print(f"   P(BatteryType): {result['BatteryType']}")
    
    # 带证据的查询
    result = bn.query(["Capacity"], evidence={"BatteryType": "Solid"})
    print(f"   P(Capacity | BatteryType=Solid): {result['Capacity']}")
    
    # 采样
    print("\n3. 从网络采样 (前5个样本):")
    samples = bn.sample(5)
    for i, sample in enumerate(samples, 1):
        print(f"   样本{i}: {sample}")
    
    print("\n4. 结构学习演示:")
    # 生成合成数据
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟数据
    battery_type = np.random.choice(["Li-ion", "Li-poly", "Solid"], n_samples, p=[0.5, 0.3, 0.2])
    temperature = np.random.choice(["Low", "Medium", "High"], n_samples, p=[0.2, 0.5, 0.3])
    
    # 根据结构生成依赖关系
    capacity = []
    for bt, temp in zip(battery_type, temperature):
        if bt == "Li-ion":
            cap = np.random.choice(["Low", "Medium", "High"], p=[0.1, 0.35, 0.55])
        elif bt == "Li-poly":
            cap = np.random.choice(["Low", "Medium", "High"], p=[0.2, 0.45, 0.35])
        else:
            cap = np.random.choice(["Low", "Medium", "High"], p=[0.25, 0.5, 0.25])
        capacity.append(cap)
    
    data = np.column_stack([battery_type, temperature, capacity])
    var_names = ["BatteryType", "Temperature", "Capacity"]
    
    # 学习结构
    learner = BNLearner()
    learner.nodes = bn.nodes  # 传递节点信息
    learned_bn = learner.fit_structure(data, var_names, method="k2", max_parents=2)
    
    print(f"   原始边数: 2 (BatteryType->Capacity, Temperature->Capacity)")
    print(f"   学习到的边数: {len(learned_bn.edges)}")
    print(f"   学习到的边: {learned_bn.edges}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
