"""
DFT-LAMMPS 能力图谱系统
=======================
模块功能→能力节点→组合路径

构建能力图谱，支持智能路径搜索和模块组合推荐。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import heapq
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union
)
from functools import total_ordering

from .module_registry import (
    ModuleRegistry, Capability, CapabilityType, RegisteredModule
)


logger = logging.getLogger("capability_graph")


class NodeType(Enum):
    """节点类型"""
    CAPABILITY = "capability"      # 能力节点
    MODULE = "module"              # 模块节点
    DATA = "data"                  # 数据节点
    OPERATION = "operation"        # 操作节点
    TOPIC = "topic"                # 课题节点


class EdgeType(Enum):
    """边类型"""
    PROVIDES = "provides"          # 模块提供能力
    REQUIRES = "requires"          # 能力需要模块
    TRANSFORMS = "transforms"      # 能力转换数据
    DEPENDS = "depends"            # 依赖关系
    SEQUENCES = "sequences"        # 序列关系
    ALTERNATIVE = "alternative"    # 替代关系


@dataclass
class CapabilityNode:
    """能力图谱节点"""
    id: str                                 # 节点ID
    node_type: NodeType                     # 节点类型
    name: str                               # 显示名称
    description: str = ""                   # 描述
    
    # 类型特定的属性
    capability_type: Optional[CapabilityType] = None  # 能力类型
    module_name: Optional[str] = None       # 关联模块名
    data_schema: Optional[Dict[str, Any]] = None      # 数据schema
    
    # 运行时属性
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, CapabilityNode):
            return self.id == other.id
        return False


@dataclass
class CapabilityEdge:
    """能力图谱边"""
    source: str                             # 源节点ID
    target: str                             # 目标节点ID
    edge_type: EdgeType                     # 边类型
    weight: float = 1.0                     # 权重（越小越优先）
    
    # 转换规则
    transform_rule: Optional[Callable] = None
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def __hash__(self) -> int:
        return hash((self.source, self.target, self.edge_type))
    
    def __eq__(self, other) -> bool:
        if isinstance(other, CapabilityEdge):
            return (self.source, self.target, self.edge_type) == \
                   (other.source, other.target, other.edge_type)
        return False


@dataclass
@total_ordering
class PathNode:
    """路径节点（用于搜索）"""
    node_id: str
    g_cost: float = 0.0           # 实际代价
    h_cost: float = 0.0           # 启发式代价
    parent: Optional[PathNode] = None
    
    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost
    
    def __lt__(self, other: PathNode) -> bool:
        return self.f_cost < other.f_cost
    
    def __eq__(self, other) -> bool:
        if isinstance(other, PathNode):
            return self.node_id == other.node_id
        return False
    
    def __hash__(self) -> int:
        return hash(self.node_id)


@dataclass
class CapabilityPath:
    """能力路径"""
    start: str                              # 起始节点
    end: str                                # 目标节点
    nodes: List[str] = field(default_factory=list)      # 路径节点
    edges: List[CapabilityEdge] = field(default_factory=list)  # 边
    total_cost: float = 0.0                 # 总代价
    estimated_quality: float = 1.0          # 估计质量
    
    def to_workflow_steps(self) -> List[Dict[str, Any]]:
        """转换为工作流步骤"""
        steps = []
        for i, node_id in enumerate(self.nodes):
            step = {
                "step": i,
                "node": node_id,
                "operations": []
            }
            if i < len(self.edges):
                edge = self.edges[i]
                step["operation"] = edge.edge_type.value
                step["description"] = edge.description
            steps.append(step)
        return steps
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "nodes": self.nodes,
            "total_cost": self.total_cost,
            "estimated_quality": self.estimated_quality,
            "steps": self.to_workflow_steps()
        }


class CapabilityGraph:
    """
    能力图谱
    
    管理模块能力之间的关系，支持路径搜索和智能推荐。
    
    Example:
        graph = CapabilityGraph()
        
        # 添加节点
        graph.add_node(CapabilityNode("vasp_energy", NodeType.CAPABILITY, "DFT能量计算"))
        
        # 添加边
        graph.add_edge(CapabilityEdge("vasp_module", "vasp_energy", EdgeType.PROVIDES))
        
        # 搜索路径
        paths = graph.find_paths("structure_input", "energy_output")
    """
    
    def __init__(self):
        self._nodes: Dict[str, CapabilityNode] = {}
        self._edges: Dict[str, Set[CapabilityEdge]] = defaultdict(set)
        self._reverse_edges: Dict[str, Set[CapabilityEdge]] = defaultdict(set)
        self._capability_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 缓存
        self._path_cache: Dict[Tuple[str, str], List[CapabilityPath]] = {}
        self._cache_enabled: bool = True
    
    def add_node(self, node: CapabilityNode) -> bool:
        """添加节点"""
        if node.id in self._nodes:
            logger.warning(f"Node {node.id} already exists")
            return False
        
        self._nodes[node.id] = node
        
        # 更新索引
        if node.capability_type:
            self._capability_index[node.capability_type.value].add(node.id)
        
        self._invalidate_cache()
        logger.debug(f"Added node: {node.id}")
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """移除节点"""
        if node_id not in self._nodes:
            return False
        
        # 移除相关边
        for edge in list(self._edges.get(node_id, [])):
            self.remove_edge(edge.source, edge.target, edge.edge_type)
        
        for edge in list(self._reverse_edges.get(node_id, [])):
            self.remove_edge(edge.source, edge.target, edge.edge_type)
        
        node = self._nodes.pop(node_id)
        
        # 更新索引
        if node.capability_type:
            self._capability_index[node.capability_type.value].discard(node_id)
        
        self._invalidate_cache()
        return True
    
    def add_edge(self, edge: CapabilityEdge) -> bool:
        """添加边"""
        if edge.source not in self._nodes or edge.target not in self._nodes:
            logger.warning(f"Cannot add edge: nodes not found")
            return False
        
        self._edges[edge.source].add(edge)
        self._reverse_edges[edge.target].add(edge)
        
        self._invalidate_cache((edge.source, edge.target))
        logger.debug(f"Added edge: {edge.source} -> {edge.target}")
        return True
    
    def remove_edge(
        self, 
        source: str, 
        target: str, 
        edge_type: EdgeType
    ) -> bool:
        """移除边"""
        edge_to_remove = None
        for edge in self._edges.get(source, []):
            if edge.target == target and edge.edge_type == edge_type:
                edge_to_remove = edge
                break
        
        if edge_to_remove:
            self._edges[source].discard(edge_to_remove)
            self._reverse_edges[target].discard(edge_to_remove)
            self._invalidate_cache((source, target))
            return True
        return False
    
    def get_node(self, node_id: str) -> Optional[CapabilityNode]:
        """获取节点"""
        return self._nodes.get(node_id)
    
    def get_neighbors(
        self, 
        node_id: str, 
        edge_type: Optional[EdgeType] = None
    ) -> List[Tuple[str, CapabilityEdge]]:
        """获取邻居节点"""
        edges = self._edges.get(node_id, [])
        
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        
        return [(e.target, e) for e in edges]
    
    def get_predecessors(
        self, 
        node_id: str, 
        edge_type: Optional[EdgeType] = None
    ) -> List[Tuple[str, CapabilityEdge]]:
        """获取前驱节点"""
        edges = self._reverse_edges.get(node_id, [])
        
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        
        return [(e.source, e) for e in edges]
    
    def find_nodes(
        self,
        node_type: Optional[NodeType] = None,
        capability_type: Optional[CapabilityType] = None,
        tags: Optional[List[str]] = None,
        keyword: Optional[str] = None
    ) -> List[CapabilityNode]:
        """搜索节点"""
        results = list(self._nodes.values())
        
        if node_type:
            results = [n for n in results if n.node_type == node_type]
        
        if capability_type:
            results = [n for n in results if n.capability_type == capability_type]
        
        if tags:
            results = [
                n for n in results 
                if any(tag in n.tags for tag in tags)
            ]
        
        if keyword:
            keyword_lower = keyword.lower()
            results = [
                n for n in results
                if (keyword_lower in n.name.lower() or
                    keyword_lower in n.description.lower())
            ]
        
        return results
    
    def find_paths(
        self,
        start: str,
        end: str,
        max_paths: int = 3,
        max_depth: int = 10,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[CapabilityPath]:
        """
        查找路径（A*算法）
        
        Args:
            start: 起始节点ID
            end: 目标节点ID
            max_paths: 最大路径数
            max_depth: 最大深度
            constraints: 约束条件
        
        Returns:
            按代价排序的路径列表
        """
        # 检查缓存
        cache_key = (start, end)
        if self._cache_enabled and cache_key in self._path_cache:
            return self._path_cache[cache_key][:max_paths]
        
        if start not in self._nodes or end not in self._nodes:
            logger.warning(f"Start or end node not found: {start} -> {end}")
            return []
        
        # A*搜索
        paths = self._astar_search(
            start, end, max_paths, max_depth, constraints or {}
        )
        
        # 缓存结果
        if self._cache_enabled:
            self._path_cache[cache_key] = paths
        
        return paths
    
    def find_alternative_paths(
        self,
        path: CapabilityPath,
        max_alternatives: int = 2
    ) -> List[CapabilityPath]:
        """查找替代路径"""
        alternatives = []
        
        # 对路径中的每个节点，尝试找替代
        for i, node_id in enumerate(path.nodes[1:-1], 1):
            # 找同类型的替代节点
            node = self._nodes.get(node_id)
            if not node:
                continue
            
            candidates = self.find_nodes(
                node_type=node.node_type,
                capability_type=node.capability_type
            )
            
            for candidate in candidates:
                if candidate.id == node_id:
                    continue
                
                # 尝试替换该节点
                prefix = path.nodes[:i]
                suffix = path.nodes[i+1:]
                
                # 找从前缀最后一个到候选节点的路径
                prefix_path = self.find_paths(
                    prefix[-1] if prefix else path.start,
                    candidate.id,
                    max_paths=1
                )
                
                if prefix_path:
                    # 找从候选节点到后缀的路径
                    suffix_path = self.find_paths(
                        candidate.id,
                        suffix[0] if suffix else path.end,
                        max_paths=1
                    )
                    
                    if suffix_path:
                        # 组合路径
                        combined = self._combine_paths(
                            prefix_path[0], suffix_path[0]
                        )
                        alternatives.append(combined)
                        
                        if len(alternatives) >= max_alternatives:
                            return alternatives
        
        return alternatives
    
    def get_execution_plan(
        self,
        path: CapabilityPath
    ) -> List[Dict[str, Any]]:
        """
        将路径转换为执行计划
        
        解析路径中的模块依赖关系，生成可执行的工作流步骤
        """
        plan = []
        
        for i, node_id in enumerate(path.nodes):
            node = self._nodes.get(node_id)
            if not node:
                continue
            
            step = {
                "step": i,
                "node_id": node_id,
                "node_name": node.name,
                "node_type": node.node_type.value,
                "description": node.description,
                "operations": []
            }
            
            # 收集相关操作
            if i < len(path.edges):
                edge = path.edges[i]
                step["operations"].append({
                    "type": edge.edge_type.value,
                    "description": edge.description,
                    "weight": edge.weight
                })
            
            # 如果是模块节点，添加模块信息
            if node.node_type == NodeType.MODULE and node.module_name:
                step["module_name"] = node.module_name
            
            plan.append(step)
        
        return plan
    
    def recommend_combinations(
        self,
        required_capabilities: List[str],
        preferences: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        推荐模块组合
        
        根据所需能力推荐最优的模块组合方案
        """
        preferences = preferences or {}
        
        # 收集提供所需能力的模块
        candidates = []
        for cap_id in required_capabilities:
            providers = self._find_capability_providers(cap_id)
            candidates.append({
                "capability": cap_id,
                "providers": providers
            })
        
        # 生成组合方案（笛卡尔积）
        from itertools import product
        
        provider_lists = [
            [(c["capability"], p) for p in c["providers"]]
            for c in candidates
        ]
        
        combinations = []
        for combo in product(*provider_lists):
            # 评估组合
            score = self._evaluate_combination(combo, preferences)
            combinations.append({
                "combination": dict(combo),
                "score": score,
                "modules": list(set(p for _, p in combo)),
                "capabilities": [c for c, _ in combo]
            })
        
        # 按分数排序
        combinations.sort(key=lambda x: x["score"], reverse=True)
        
        return combinations[:10]  # 返回前10个
    
    def analyze_coverage(
        self,
        required_capabilities: List[str]
    ) -> Dict[str, Any]:
        """分析能力覆盖情况"""
        covered = []
        missing = []
        partial = []
        
        for cap_id in required_capabilities:
            providers = self._find_capability_providers(cap_id)
            if len(providers) >= 2:
                covered.append({
                    "capability": cap_id,
                    "providers": providers
                })
            elif len(providers) == 1:
                partial.append({
                    "capability": cap_id,
                    "provider": providers[0]
                })
            else:
                missing.append(cap_id)
        
        coverage_rate = len(covered) / len(required_capabilities) if required_capabilities else 0
        
        return {
            "coverage_rate": coverage_rate,
            "covered": covered,
            "partial": partial,
            "missing": missing,
            "recommendations": self._generate_recommendations(missing)
        }
    
    def export_graph(self, format: str = "json") -> str:
        """导出图谱"""
        data = {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.node_type.value,
                    "name": n.name,
                    "description": n.description,
                    "capability_type": n.capability_type.value if n.capability_type else None,
                    "module_name": n.module_name,
                    "tags": n.tags
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.edge_type.value,
                    "weight": e.weight,
                    "description": e.description
                }
                for edges in self._edges.values()
                for e in edges
            ]
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_graph(self, data: str, format: str = "json") -> None:
        """导入图谱"""
        if format == "json":
            parsed = json.loads(data)
            
            # 清空现有图谱
            self._nodes.clear()
            self._edges.clear()
            self._reverse_edges.clear()
            
            # 添加节点
            for node_data in parsed.get("nodes", []):
                node = CapabilityNode(
                    id=node_data["id"],
                    node_type=NodeType(node_data["type"]),
                    name=node_data["name"],
                    description=node_data.get("description", ""),
                    capability_type=CapabilityType(node_data["capability_type"]) if node_data.get("capability_type") else None,
                    module_name=node_data.get("module_name"),
                    tags=node_data.get("tags", [])
                )
                self.add_node(node)
            
            # 添加边
            for edge_data in parsed.get("edges", []):
                edge = CapabilityEdge(
                    source=edge_data["source"],
                    target=edge_data["target"],
                    edge_type=EdgeType(edge_data["type"]),
                    weight=edge_data.get("weight", 1.0),
                    description=edge_data.get("description", "")
                )
                self.add_edge(edge)
            
            logger.info(f"Imported graph with {len(self._nodes)} nodes and {sum(len(e) for e in self._edges.values())} edges")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)
        
        for node in self._nodes.values():
            node_types[node.node_type.value] += 1
        
        for edges in self._edges.values():
            for edge in edges:
                edge_types[edge.edge_type.value] += 1
        
        return {
            "total_nodes": len(self._nodes),
            "total_edges": sum(len(e) for e in self._edges.values()),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "is_connected": self._check_connectivity()
        }
    
    def build_from_registry(
        self, 
        registry: Optional[ModuleRegistry] = None
    ) -> None:
        """
        从模块注册中心构建能力图谱
        
        自动分析已注册模块，构建完整的能力图谱
        """
        if registry is None:
            registry = ModuleRegistry.get_instance()
        
        modules = registry.get_all_modules()
        
        for mod in modules:
            # 添加模块节点
            module_node = CapabilityNode(
                id=f"module_{mod.metadata.name}",
                node_type=NodeType.MODULE,
                name=mod.metadata.name,
                description=mod.metadata.description,
                module_name=mod.metadata.name,
                tags=mod.metadata.keywords
            )
            self.add_node(module_node)
            
            # 添加能力节点和关系
            for cap in mod.capabilities:
                cap_id = f"cap_{cap.name}"
                
                # 添加能力节点（如果不存在）
                if cap_id not in self._nodes:
                    cap_node = CapabilityNode(
                        id=cap_id,
                        node_type=NodeType.CAPABILITY,
                        name=cap.name,
                        description=cap.description,
                        capability_type=cap.capability_type,
                        tags=cap.tags
                    )
                    self.add_node(cap_node)
                
                # 添加提供关系
                self.add_edge(CapabilityEdge(
                    source=module_node.id,
                    target=cap_id,
                    edge_type=EdgeType.PROVIDES,
                    weight=0.5,
                    description=f"{mod.metadata.name} provides {cap.name}"
                ))
            
            # 添加依赖关系
            for dep in mod.dependencies:
                dep_module_id = f"module_{dep.module_name}"
                if dep_module_id in self._nodes:
                    self.add_edge(CapabilityEdge(
                        source=module_node.id,
                        target=dep_module_id,
                        edge_type=EdgeType.DEPENDS,
                        weight=0.3,
                        description=f"Depends on {dep.module_name}"
                    ))
        
        logger.info(f"Built graph from {len(modules)} modules")
    
    def _astar_search(
        self,
        start: str,
        end: str,
        max_paths: int,
        max_depth: int,
        constraints: Dict[str, Any]
    ) -> List[CapabilityPath]:
        """A*路径搜索"""
        paths = []
        visited = set()
        
        # 启发式函数
        def heuristic(node_id: str) -> float:
            # 简单的启发式：基于图距离估计
            return 0.0  # 可扩展为更复杂的启发式
        
        # 优先队列
        open_set: List[PathNode] = []
        heapq.heappush(open_set, PathNode(start, 0, heuristic(start)))
        
        # 记录每个节点的最优路径
        best_paths: Dict[str, List[Tuple[float, List[str], List[CapabilityEdge]]]] = defaultdict(list)
        best_paths[start].append((0, [start], []))
        
        while open_set and len(paths) < max_paths:
            current = heapq.heappop(open_set)
            
            if current.node_id == end:
                # 找到路径
                for cost, nodes, edges in best_paths[end]:
                    if len(paths) < max_paths:
                        paths.append(CapabilityPath(
                            start=start,
                            end=end,
                            nodes=nodes,
                            edges=edges,
                            total_cost=cost
                        ))
                continue
            
            if current.node_id in visited:
                continue
            
            if len(best_paths[current.node_id][0][1]) > max_depth:
                continue
            
            visited.add(current.node_id)
            
            # 扩展邻居
            for neighbor_id, edge in self.get_neighbors(current.node_id):
                if neighbor_id in visited:
                    continue
                
                # 检查约束
                if not self._check_constraints(edge, constraints):
                    continue
                
                new_cost = current.g_cost + edge.weight
                new_nodes = best_paths[current.node_id][0][1] + [neighbor_id]
                new_edges = best_paths[current.node_id][0][2] + [edge]
                
                # 更新最佳路径
                existing = best_paths[neighbor_id]
                if not existing or new_cost < existing[0][0]:
                    best_paths[neighbor_id] = [(new_cost, new_nodes, new_edges)]
                    heapq.heappush(
                        open_set,
                        PathNode(neighbor_id, new_cost, heuristic(neighbor_id))
                    )
        
        # 按代价排序
        paths.sort(key=lambda p: p.total_cost)
        return paths
    
    def _check_constraints(
        self, 
        edge: CapabilityEdge, 
        constraints: Dict[str, Any]
    ) -> bool:
        """检查边是否满足约束"""
        # 检查条件
        if edge.condition and not edge.condition(constraints):
            return False
        
        # 检查黑名单
        blacklist = constraints.get('blacklist', [])
        if edge.target in blacklist or edge.source in blacklist:
            return False
        
        # 检查必须的节点类型
        required_types = constraints.get('required_node_types', [])
        if required_types:
            target_node = self._nodes.get(edge.target)
            if target_node and target_node.node_type.value not in required_types:
                return False
        
        return True
    
    def _combine_paths(
        self, 
        path1: CapabilityPath, 
        path2: CapabilityPath
    ) -> CapabilityPath:
        """组合两条路径"""
        # 确保path1.end == path2.start
        if path1.end != path2.start:
            raise ValueError("Cannot combine non-continuous paths")
        
        combined_nodes = path1.nodes + path2.nodes[1:]
        combined_edges = path1.edges + path2.edges
        
        return CapabilityPath(
            start=path1.start,
            end=path2.end,
            nodes=combined_nodes,
            edges=combined_edges,
            total_cost=path1.total_cost + path2.total_cost
        )
    
    def _find_capability_providers(self, capability_id: str) -> List[str]:
        """查找提供某能力的模块"""
        providers = []
        
        # 查找PROVIDES边
        for edge in self._reverse_edges.get(capability_id, []):
            if edge.edge_type == EdgeType.PROVIDES:
                providers.append(edge.source)
        
        return providers
    
    def _evaluate_combination(
        self,
        combination: List[Tuple[str, str]],
        preferences: Dict[str, Any]
    ) -> float:
        """评估组合方案"""
        score = 1.0
        
        # 模块数量惩罚（越少越好）
        unique_modules = len(set(p for _, p in combination))
        score -= 0.1 * unique_modules
        
        # 偏好匹配
        preferred_modules = preferences.get('preferred_modules', [])
        for _, module in combination:
            if module in preferred_modules:
                score += 0.2
        
        # 避免黑名单
        blacklist = preferences.get('blacklist', [])
        for _, module in combination:
            if module in blacklist:
                score -= 1.0
        
        return max(0, score)
    
    def _generate_recommendations(
        self, 
        missing_capabilities: List[str]
    ) -> List[str]:
        """为缺失的能力生成推荐"""
        recommendations = []
        
        for cap in missing_capabilities:
            # 查找相似能力
            similar = self.find_nodes(keyword=cap.split("_")[0] if "_" in cap else cap)
            if similar:
                recommendations.append(
                    f"Consider using {similar[0].name} as alternative to {cap}"
                )
            else:
                recommendations.append(
                    f"No alternative found for {cap}, consider implementing a new module"
                )
        
        return recommendations
    
    def _invalidate_cache(
        self, 
        key: Optional[Tuple[str, str]] = None
    ) -> None:
        """使缓存失效"""
        if key:
            self._path_cache.pop(key, None)
        else:
            self._path_cache.clear()
    
    def _check_connectivity(self) -> bool:
        """检查图谱连通性"""
        if not self._nodes:
            return True
        
        # BFS检查
        start = next(iter(self._nodes.keys()))
        visited = {start}
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            for neighbor, _ in self.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(self._nodes)


# 便捷函数
def get_global_graph() -> CapabilityGraph:
    """获取全局能力图谱实例"""
    if not hasattr(get_global_graph, '_instance'):
        get_global_graph._instance = CapabilityGraph()
    return get_global_graph._instance


def build_graph_from_registry(
    registry: Optional[ModuleRegistry] = None
) -> CapabilityGraph:
    """从注册中心构建图谱"""
    graph = get_global_graph()
    graph.build_from_registry(registry)
    return graph


def find_module_combination(
    required_capabilities: List[str],
    **kwargs
) -> List[Dict[str, Any]]:
    """便捷函数：查找模块组合"""
    return get_global_graph().recommend_combinations(required_capabilities, kwargs)