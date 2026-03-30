"""
Neo4j Graph Database - Neo4j图数据库集成
======================================
提供知识图谱存储、查询和推理功能。

特性：
- 节点和关系CRUD操作
- Cypher查询执行
- 路径查询和图遍历
- 图算法集成 (GDS)
- 事务支持
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Neo4j配置"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_pool_size: int = 100
    connection_timeout: int = 30
    encrypted: bool = True
    trust: str = "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "auth": (self.username, self.password),
            "max_connection_pool_size": self.max_connection_pool_size,
            "connection_timeout": self.connection_timeout,
            "encrypted": self.encrypted
        }


@dataclass
class NodeSpec:
    """节点规格"""
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 自动添加创建时间
        if "created_at" not in self.properties:
            self.properties["created_at"] = datetime.now().isoformat()
    
    def to_cypher(self, var_name: str = "n") -> Tuple[str, Dict[str, Any]]:
        """转换为Cypher语句片段"""
        labels_str = ":" + ":".join(self.labels) if self.labels else ""
        props_str = ", ".join([f"{k}: ${k}" for k in self.properties.keys()])
        cypher = f"({var_name}{labels_str} {{{props_str}}})"
        return cypher, self.properties


@dataclass
class RelationSpec:
    """关系规格"""
    rel_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if "created_at" not in self.properties:
            self.properties["created_at"] = datetime.now().isoformat()
    
    def to_cypher(self, var_name: str = "r") -> Tuple[str, Dict[str, Any]]:
        """转换为Cypher语句片段"""
        props_str = ", ".join([f"{k}: ${k}" for k in self.properties.keys()])
        if props_str:
            cypher = f"[{var_name}:{self.rel_type} {{{props_str}}}]"
        else:
            cypher = f"[{var_name}:{self.rel_type}]"
        return cypher, self.properties


@dataclass
class GraphPattern:
    """图模式"""
    start_node: NodeSpec
    relation: RelationSpec
    end_node: NodeSpec
    direction: str = "->"  # ->, <-, -
    
    def to_cypher(self, start_var: str = "a", rel_var: str = "r", end_var: str = "b") -> Tuple[str, Dict[str, Any]]:
        """转换为Cypher模式"""
        start_cypher, start_props = self.start_node.to_cypher(start_var)
        rel_cypher, rel_props = self.relation.to_cypher(rel_var)
        end_cypher, end_props = self.end_node.to_cypher(end_var)
        
        # 重命名参数以避免冲突
        start_props = {f"start_{k}": v for k, v in start_props.items()}
        rel_props = {f"rel_{k}": v for k, v in rel_props.items()}
        end_props = {f"end_{k}": v for k, v in end_props.items()}
        
        # 重建Cypher片段
        start_labels = ":" + ":".join(self.start_node.labels) if self.start_node.labels else ""
        start_props_str = ", ".join([f"{k}: $start_{k}" for k in self.start_node.properties.keys()])
        start_cypher = f"({start_var}{start_labels} {{{start_props_str}}})"
        
        end_labels = ":" + ":".join(self.end_node.labels) if self.end_node.labels else ""
        end_props_str = ", ".join([f"{k}: $end_{k}" for k in self.end_node.properties.keys()])
        end_cypher = f"({end_var}{end_labels} {{{end_props_str}}})"
        
        rel_props_str = ", ".join([f"{k}: $rel_{k}" for k in self.relation.properties.keys()])
        if rel_props_str:
            rel_cypher = f"[{rel_var}:{self.relation.rel_type} {{{rel_props_str}}}]"
        else:
            rel_cypher = f"[{rel_var}:{self.relation.rel_type}]"
        
        # 构建方向
        if self.direction == "->":
            pattern = f"{start_cypher}-[{rel_cypher}]->{end_cypher}"
        elif self.direction == "<-":
            pattern = f"{start_cypher}<-[{rel_cypher}]-{end_cypher}"
        else:
            pattern = f"{start_cypher}-[{rel_cypher}]-{end_cypher}"
        
        all_props = {**start_props, **rel_props, **end_props}
        return pattern, all_props


@dataclass
class GraphQuery:
    """图查询"""
    match_clause: str
    where_clause: Optional[str] = None
    return_clause: str = "*"
    order_by: Optional[str] = None
    limit: Optional[int] = None
    skip: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_cypher(self) -> str:
        """构建完整Cypher查询"""
        parts = [f"MATCH {self.match_clause}"]
        
        if self.where_clause:
            parts.append(f"WHERE {self.where_clause}")
        
        parts.append(f"RETURN {self.return_clause}")
        
        if self.order_by:
            parts.append(f"ORDER BY {self.order_by}")
        
        if self.skip:
            parts.append(f"SKIP {self.skip}")
        
        if self.limit:
            parts.append(f"LIMIT {self.limit}")
        
        return " ".join(parts)


@dataclass
class PathQuery:
    """路径查询"""
    start_node_id: str
    end_node_id: Optional[str] = None
    min_depth: int = 1
    max_depth: int = 4
    relationship_types: Optional[List[str]] = None
    path_algorithm: str = "shortestPath"  # shortestPath, allShortestPaths, apoc.path.expand
    
    def to_cypher(self) -> str:
        """构建路径查询Cypher"""
        rel_filter = "|".join(self.relationship_types) if self.relationship_types else ""
        
        if self.end_node_id:
            # 两点间最短路径
            return f"""
                MATCH p = {self.path_algorithm}((a)-[{rel_filter}*1..{self.max_depth}]-(b))
                WHERE id(a) = $start_id AND id(b) = $end_id
                RETURN p
            """
        else:
            # 从起点展开
            return f"""
                MATCH path = (start)-[{rel_filter}*{self.min_depth}..{self.max_depth}]->(end)
                WHERE id(start) = $start_id
                RETURN path, end
            """


@dataclass
class GraphPath:
    """图路径"""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "relationships": self.relationships,
            "length": self.length
        }


@dataclass
class GraphMetrics:
    """图指标"""
    node_count: int = 0
    relationship_count: int = 0
    label_counts: Dict[str, int] = field(default_factory=dict)
    relationship_type_counts: Dict[str, int] = field(default_factory=dict)
    density: float = 0.0
    avg_degree: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": self.node_count,
            "relationship_count": self.relationship_count,
            "label_counts": self.label_counts,
            "relationship_type_counts": self.relationship_type_counts,
            "density": self.density,
            "avg_degree": self.avg_degree
        }


class Neo4jGraphDB:
    """
    Neo4j图数据库客户端
    
    提供知识图谱的存储、查询和推理功能。
    """
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        self.config = config or Neo4jConfig()
        self._driver = None
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> bool:
        """连接到Neo4j"""
        try:
            from neo4j import GraphDatabase
            
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout,
                encrypted=self.config.encrypted
            )
            
            # 验证连接
            self._driver.verify_connectivity()
            self._connected = True
            
            logger.info(f"Connected to Neo4j at {self.config.uri}")
            return True
            
        except ImportError:
            logger.error("neo4j not installed. Run: pip install neo4j")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._connected = False
            logger.info("Disconnected from Neo4j")
    
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        执行Cypher查询
        
        Args:
            query: Cypher查询语句
            parameters: 查询参数
            database: 目标数据库
            
        Returns:
            查询结果列表
        """
        if not self._connected:
            raise RuntimeError("Not connected to Neo4j")
        
        try:
            with self._driver.session(database=database or self.config.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行写操作
        
        Args:
            query: Cypher写语句
            parameters: 查询参数
            database: 目标数据库
            
        Returns:
            操作统计信息
        """
        if not self._connected:
            raise RuntimeError("Not connected to Neo4j")
        
        try:
            with self._driver.session(database=database or self.config.database) as session:
                result = session.execute_write(self._do_write, query, parameters or {})
                return result
        except Exception as e:
            logger.error(f"Write failed: {e}")
            raise
    
    def _do_write(self, tx, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行写事务"""
        result = tx.run(query, parameters)
        summary = result.consume()
        return {
            "nodes_created": summary.counters.nodes_created,
            "nodes_deleted": summary.counters.nodes_deleted,
            "relationships_created": summary.counters.relationships_created,
            "relationships_deleted": summary.counters.relationships_deleted,
            "properties_set": summary.counters.properties_set,
            "labels_added": summary.counters.labels_added,
            "labels_removed": summary.counters.labels_removed
        }
    
    # ==================== 节点操作 ====================
    
    def create_node(self, spec: NodeSpec) -> str:
        """创建节点"""
        cypher, params = spec.to_cypher("n")
        query = f"CREATE {cypher} RETURN id(n) as node_id"
        
        result = self.execute_write(query, params)
        return result
    
    def create_nodes(self, specs: List[NodeSpec]) -> List[str]:
        """批量创建节点"""
        node_ids = []
        for spec in specs:
            result = self.create_node(spec)
            node_ids.append(result)
        return node_ids
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取节点"""
        query = "MATCH (n) WHERE id(n) = $node_id RETURN n"
        results = self.execute_query(query, {"node_id": int(node_id)})
        return results[0]["n"] if results else None
    
    def get_nodes_by_label(
        self,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """根据标签获取节点"""
        if properties:
            props_str = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"""
                MATCH (n:{label})
                WHERE {props_str}
                RETURN n
                LIMIT {limit}
            """
            params = properties
        else:
            query = f"MATCH (n:{label}) RETURN n LIMIT {limit}"
            params = {}
        
        results = self.execute_query(query, params)
        return [r["n"] for r in results]
    
    def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """更新节点属性"""
        if merge:
            set_clause = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"""
                MATCH (n)
                WHERE id(n) = $node_id
                SET {set_clause}
                RETURN id(n)
            """
        else:
            # 完全替换属性
            query = """
                MATCH (n)
                WHERE id(n) = $node_id
                SET n = $properties
                RETURN id(n)
            """
            properties = {"properties": properties}
        
        properties["node_id"] = int(node_id)
        self.execute_write(query, properties)
        return True
    
    def delete_node(self, node_id: str, detach: bool = True) -> bool:
        """删除节点"""
        detach_keyword = "DETACH" if detach else ""
        query = f"""
            MATCH (n)
            WHERE id(n) = $node_id
            {detach_keyword} DELETE n
        """
        self.execute_write(query, {"node_id": int(node_id)})
        return True
    
    # ==================== 关系操作 ====================
    
    def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        spec: RelationSpec
    ) -> str:
        """创建关系"""
        rel_cypher, params = spec.to_cypher("r")
        
        query = f"""
            MATCH (a), (b)
            WHERE id(a) = $from_id AND id(b) = $to_id
            CREATE (a)-{rel_cypher}->(b)
            RETURN id(r) as rel_id
        """
        
        # 重命名参数
        params = {f"rel_{k}": v for k, v in params.items()}
        params["from_id"] = int(from_node_id)
        params["to_id"] = int(to_node_id)
        
        result = self.execute_write(query, params)
        return result
    
    def get_relationships(
        self,
        node_id: str,
        direction: str = "both"  # in, out, both
    ) -> List[Dict[str, Any]]:
        """获取节点的关系"""
        if direction == "out":
            query = """
                MATCH (n)-[r]->(m)
                WHERE id(n) = $node_id
                RETURN r, m as connected_node
            """
        elif direction == "in":
            query = """
                MATCH (n)<-[r]-(m)
                WHERE id(n) = $node_id
                RETURN r, m as connected_node
            """
        else:
            query = """
                MATCH (n)-[r]-(m)
                WHERE id(n) = $node_id
                RETURN r, m as connected_node
            """
        
        return self.execute_query(query, {"node_id": int(node_id)})
    
    def delete_relationship(self, rel_id: str) -> bool:
        """删除关系"""
        query = """
            MATCH ()-[r]->()
            WHERE id(r) = $rel_id
            DELETE r
        """
        self.execute_write(query, {"rel_id": int(rel_id)})
        return True
    
    # ==================== 模式匹配 ====================
    
    def match_pattern(
        self,
        pattern: GraphPattern,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """匹配图模式"""
        pattern_cypher, params = pattern.to_cypher()
        query = f"""
            MATCH {pattern_cypher}
            RETURN a, r, b
            LIMIT {limit}
        """
        return self.execute_query(query, params)
    
    def find_paths(self, query: PathQuery) -> List[GraphPath]:
        """查找路径"""
        cypher = query.to_cypher()
        params = {"start_id": int(query.start_node_id)}
        
        if query.end_node_id:
            params["end_id"] = int(query.end_node_id)
        
        results = self.execute_query(cypher, params)
        
        paths = []
        for result in results:
            if "p" in result:
                path_data = result["p"]
                path = GraphPath(
                    nodes=path_data.get("nodes", []),
                    relationships=path_data.get("relationships", []),
                    length=len(path_data.get("relationships", []))
                )
                paths.append(path)
        
        return paths
    
    # ==================== 图算法 ====================
    
    def page_rank(
        self,
        label: Optional[str] = None,
        relationship_type: Optional[str] = None,
        iterations: int = 20,
        damping_factor: float = 0.85
    ) -> List[Dict[str, Any]]:
        """PageRank算法"""
        if label:
            node_query = f"MATCH (n:{label}) RETURN id(n) as id"
        else:
            node_query = "MATCH (n) RETURN id(n) as id"
        
        if relationship_type:
            rel_query = f"MATCH ()-[r:{relationship_type}]->() RETURN id(startNode(r)) as source, id(endNode(r)) as target"
        else:
            rel_query = "MATCH ()-[r]->() RETURN id(startNode(r)) as source, id(endNode(r)) as target"
        
        query = f"""
            CALL gds.pageRank.stream({{
                nodeQuery: '{node_query}',
                relationshipQuery: '{rel_query}',
                iterations: {iterations},
                dampingFactor: {damping_factor}
            }})
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId) as node, score
            ORDER BY score DESC
        """
        
        return self.execute_query(query)
    
    def community_detection(
        self,
        algorithm: str = "louvain",
        label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """社区检测"""
        if algorithm == "louvain":
            proc = "gds.louvain"
        elif algorithm == "labelPropagation":
            proc = "gds.labelPropagation"
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        label_filter = f":{label}" if label else ""
        query = f"""
            CALL {proc}.stream('graph-{label or "all"}')
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId) as node, communityId
            ORDER BY communityId
        """
        
        return self.execute_query(query)
    
    def shortest_path(
        self,
        start_node_id: str,
        end_node_id: str,
        relationship_type: Optional[str] = None
    ) -> Optional[GraphPath]:
        """最短路径"""
        rel_filter = f":{relationship_type}" if relationship_type else ""
        
        query = f"""
            MATCH p = shortestPath(
                (a)-[{rel_filter}*]-(b)
            )
            WHERE id(a) = $start_id AND id(b) = $end_id
            RETURN p
        """
        
        results = self.execute_query(query, {
            "start_id": int(start_node_id),
            "end_id": int(end_node_id)
        })
        
        if results and "p" in results[0]:
            path_data = results[0]["p"]
            return GraphPath(
                nodes=path_data.get("nodes", []),
                relationships=path_data.get("relationships", []),
                length=len(path_data.get("relationships", []))
            )
        return None
    
    def similarity(
        self,
        node_id: str,
        label: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """节点相似度 (基于共同邻居)"""
        label_filter = f":{label}" if label else ""
        
        query = f"""
            MATCH (n{label_filter})-[]-(neighbor)
            WHERE id(n) = $node_id
            WITH n, collect(id(neighbor)) as neighbors
            
            MATCH (m{label_filter})-[]-(neighbor)
            WHERE m <> n
            WITH n, m, neighbors, collect(id(neighbor)) as m_neighbors
            
            WITH n, m, 
                 gds.alpha.similarity.jaccard(neighbors, m_neighbors) as similarity
            WHERE similarity > 0
            RETURN m as similar_node, similarity
            ORDER BY similarity DESC
            LIMIT {top_k}
        """
        
        return self.execute_query(query, {"node_id": int(node_id)})
    
    # ==================== 统计和指标 ====================
    
    def get_metrics(self) -> GraphMetrics:
        """获取图指标"""
        # 节点数
        node_count_result = self.execute_query("MATCH (n) RETURN count(n) as count")
        node_count = node_count_result[0]["count"] if node_count_result else 0
        
        # 关系数
        rel_count_result = self.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = rel_count_result[0]["count"] if rel_count_result else 0
        
        # 标签统计
        label_result = self.execute_query("""
            MATCH (n)
            UNWIND labels(n) as label
            RETURN label, count(*) as count
        """)
        label_counts = {r["label"]: r["count"] for r in label_result}
        
        # 关系类型统计
        rel_type_result = self.execute_query("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
        """)
        rel_type_counts = {r["type"]: r["count"] for r in rel_type_result}
        
        # 计算密度和平均度
        density = 0.0
        avg_degree = 0.0
        if node_count > 0:
            avg_degree = (2 * rel_count) / node_count
            max_edges = node_count * (node_count - 1)
            if max_edges > 0:
                density = rel_count / max_edges
        
        return GraphMetrics(
            node_count=node_count,
            relationship_count=rel_count,
            label_counts=label_counts,
            relationship_type_counts=rel_type_counts,
            density=density,
            avg_degree=avg_degree
        )
    
    def clear_graph(self, confirm: bool = False):
        """清空图数据库 (危险操作!)"""
        if not confirm:
            logger.warning("Set confirm=True to clear the graph")
            return
        
        self.execute_write("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")
    
    def export_to_cypher(self, output_path: str):
        """导出图为Cypher语句"""
        query = """
            CALL apoc.export.cypher.all($path, {
                format: 'cypher-shell',
                useOptimizations: {type: 'UNWIND_BATCH', unwindBatchSize: 100}
            })
            YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
            RETURN file, nodes, relationships, time
        """
        
        result = self.execute_query(query, {"path": output_path})
        logger.info(f"Exported graph to {output_path}")
        return result
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
