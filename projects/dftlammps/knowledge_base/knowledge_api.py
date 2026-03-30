"""
Knowledge API - 知识库API
=========================
提供统一的知识库访问接口，整合所有组件。
"""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API配置"""
    # MongoDB配置
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_database: str = "dftlammps_kb"
    
    # PostgreSQL配置
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "dftlammps_kb"
    postgres_username: str = "postgres"
    postgres_password: str = ""
    
    # Neo4j配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    
    # 向量存储配置
    vector_provider: str = "local"  # pinecone, milvus, weaviate, local
    vector_dimension: int = 1536
    pinecone_api_key: Optional[str] = None
    milvus_host: str = "localhost"
    weaviate_url: str = "http://localhost:8080"
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "APIConfig":
        """从字典创建配置"""
        return cls(**config)


class DataPipeline:
    """
    数据管道
    
    处理数据从采集到存储的完整流程。
    """
    
    def __init__(self, knowledge_api: "KnowledgeAPI"):
        self.api = knowledge_api
        self._processors: List[Callable] = []
    
    def add_processor(self, processor: Callable):
        """添加处理器"""
        self._processors.append(processor)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        result = data
        for processor in self._processors:
            result = processor(result)
        return result
    
    def ingest_calculation(self, calculation: Dict[str, Any]) -> str:
        """
        摄入计算结果
        
        Args:
            calculation: 计算结果数据
            
        Returns:
            记录ID
        """
        # 处理数据
        processed = self.process(calculation)
        
        # 存储到MongoDB
        record_id = self.api.store_document("calculations", processed)
        
        # 构建知识图谱
        subgraph = self.api.builder.build_from_calculation(processed)
        
        # 存储到Neo4j
        if self.api.neo4j and self.api.neo4j.is_connected:
            for entity in subgraph["entities"]:
                from ..graph.neo4j_graph import NodeSpec
                node = NodeSpec(
                    labels=[entity["type"]],
                    properties=entity.get("properties", {})
                )
                self.api.neo4j.create_node(node)
        
        # 向量化并存储
        if self.api.vector_store and self.api.vector_store.is_connected:
            content = self._calculation_to_text(processed)
            vector = self.api.embedding_provider.embed_text(content)
            self.api.vector_store.upsert(
                vectors=[vector],
                ids=[record_id],
                metadata=[{
                    "content": content,
                    "calculation_type": processed.get("type", ""),
                    "material": processed.get("material", {}).get("formula", "")
                }]
            )
        
        # 版本控制
        version = self.api.version_control.commit(
            data=processed,
            message=f"Ingested calculation for {processed.get('material', {}).get('formula', 'unknown')}",
            calculation_type=processed.get("type", ""),
            parameters=processed.get("parameters", {}),
            results=processed.get("results", {})
        )
        
        return record_id
    
    def ingest_material(self, material: Dict[str, Any]) -> str:
        """摄入材料数据"""
        processed = self.process(material)
        record_id = self.api.store_document("materials", processed)
        
        # 添加到知识图谱
        entity = self.api.builder.add_entity(
            entity_type="Material",
            name=material.get("formula", ""),
            properties=material,
            source="ingestion"
        )
        
        return record_id
    
    def ingest_publication(self, publication: Dict[str, Any]) -> str:
        """摄入文献数据"""
        processed = self.process(publication)
        record_id = self.api.store_document("publications", processed)
        
        # 添加到知识图谱
        entity = self.api.builder.add_entity(
            entity_type="Publication",
            name=publication.get("title", ""),
            properties=publication,
            source="ingestion"
        )
        
        return record_id
    
    def _calculation_to_text(self, calculation: Dict[str, Any]) -> str:
        """将计算结果转换为文本"""
        parts = []
        
        material = calculation.get("material", {})
        if material:
            parts.append(f"Material: {material.get('formula', '')}")
            if "structure_type" in material:
                parts.append(f"Structure: {material['structure_type']}")
        
        method = calculation.get("method", {})
        if method:
            parts.append(f"Method: {method.get('name', '')}")
        
        results = calculation.get("results", {})
        if results:
            parts.append("Results:")
            for prop, value in results.items():
                parts.append(f"  {prop}: {value}")
        
        return "\n".join(parts)


class QueryBuilder:
    """
    查询构建器
    
    提供便捷的查询构建接口。
    """
    
    def __init__(self, knowledge_api: "KnowledgeAPI"):
        self.api = knowledge_api
        self._filters = []
    
    def where(self, field: str, operator: str, value: Any) -> "QueryBuilder":
        """添加过滤条件"""
        from .storage.base_storage import QueryFilter
        self._filters.append(QueryFilter(field, operator, value))
        return self
    
    def equals(self, field: str, value: Any) -> "QueryBuilder":
        """等于条件"""
        return self.where(field, "eq", value)
    
    def contains(self, field: str, value: str) -> "QueryBuilder":
        """包含条件"""
        return self.where(field, "regex", value)
    
    def greater_than(self, field: str, value: float) -> "QueryBuilder":
        """大于条件"""
        return self.where(field, "gt", value)
    
    def less_than(self, field: str, value: float) -> "QueryBuilder":
        """小于条件"""
        return self.where(field, "lt", value)
    
    def execute(self, collection: str, limit: int = 100) -> List[Dict[str, Any]]:
        """执行查询"""
        results = self.api.query_documents(collection, self._filters, limit)
        self._filters = []  # 重置
        return results
    
    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """语义搜索"""
        from .search.semantic_search import SearchQuery, SearchMode
        
        search_query = SearchQuery(
            query=query,
            mode=SearchMode[mode.upper()],
            top_k=top_k
        )
        
        return self.api.semantic_search(search_query)


class KnowledgeExporter:
    """知识导出器"""
    
    def __init__(self, knowledge_api: "KnowledgeAPI"):
        self.api = knowledge_api
    
    def export_to_json(self, output_path: str, collection: str = None):
        """导出为JSON"""
        import json
        
        if collection:
            data = self.api.query_documents(collection, limit=10000)
        else:
            # 导出所有集合
            data = self.api.builder.get_knowledge_graph()
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported to {output_path}")
    
    def export_to_csv(self, output_path: str, collection: str):
        """导出为CSV"""
        import csv
        
        data = self.api.query_documents(collection, limit=10000)
        
        if not data:
            logger.warning(f"No data in collection {collection}")
            return
        
        # 获取所有字段
        fieldnames = set()
        for item in data:
            fieldnames.update(item.keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Exported {len(data)} records to {output_path}")
    
    def export_to_neo4j_cypher(self, output_path: str):
        """导出为Neo4j Cypher语句"""
        kg = self.api.builder.get_knowledge_graph()
        
        with open(output_path, 'w') as f:
            # 生成创建节点的语句
            for entity in kg["entities"]:
                labels = ":".join([entity["type"]])
                props = ", ".join([f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}"
                                  for k, v in entity.get("properties", {}).items()])
                f.write(f"CREATE (:{labels} {{{props}}}));\n")
            
            # 生成创建关系的语句
            for relation in kg["relations"]:
                rel_type = relation["type"]
                from_id = relation["from"]
                to_id = relation["to"]
                f.write(f"""
                    MATCH (a), (b)
                    WHERE id(a) = {from_id} AND id(b) = {to_id}
                    CREATE (a)-[:{rel_type}]->(b);
                """)
        
        logger.info(f"Exported Cypher to {output_path}")
    
    def export_to_rdf(self, output_path: str):
        """导出为RDF格式"""
        # 简化的RDF导出
        kg = self.api.builder.get_knowledge_graph()
        
        with open(output_path, 'w') as f:
            f.write("@prefix kb: <http://dftlammps.org/kb#> .\n")
            f.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n\n")
            
            for entity in kg["entities"]:
                entity_uri = f"kb:entity_{entity['id']}"
                f.write(f"{entity_uri} rdf:type kb:{entity['type']} .\n")
                f.write(f'{entity_uri} kb:name "{entity["name"]}" .\n')
                
                for prop, value in entity.get("properties", {}).items():
                    if isinstance(value, str):
                        f.write(f'{entity_uri} kb:{prop} "{value}" .\n')
                    else:
                        f.write(f'{entity_uri} kb:{prop} {value} .\n')
            
            for relation in kg["relations"]:
                from_uri = f"kb:entity_{relation['from']}"
                to_uri = f"kb:entity_{relation['to']}"
                f.write(f"{from_uri} kb:{relation['type']} {to_uri} .\n")
        
        logger.info(f"Exported RDF to {output_path}")


class KnowledgeImporter:
    """知识导入器"""
    
    def __init__(self, knowledge_api: "KnowledgeAPI"):
        self.api = knowledge_api
    
    def import_from_json(self, input_path: str, collection: str = "imported") -> int:
        """从JSON导入"""
        import json
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            count = 0
            for item in data:
                self.api.store_document(collection, item)
                count += 1
            logger.info(f"Imported {count} records from {input_path}")
            return count
        else:
            self.api.store_document(collection, data)
            logger.info(f"Imported 1 record from {input_path}")
            return 1
    
    def import_from_csv(self, input_path: str, collection: str) -> int:
        """从CSV导入"""
        import csv
        
        count = 0
        with open(input_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.api.store_document(collection, dict(row))
                count += 1
        
        logger.info(f"Imported {count} records from {input_path}")
        return count
    
    def import_from_poscar(self, input_path: str) -> str:
        """从POSCAR导入结构"""
        # 简化的POSCAR解析
        material = {
            "source_file": input_path,
            "format": "POSCAR",
            "imported_at": datetime.now().isoformat()
        }
        
        return self.api.store_document("structures", material)
    
    def import_from_cif(self, input_path: str) -> str:
        """从CIF导入结构"""
        material = {
            "source_file": input_path,
            "format": "CIF",
            "imported_at": datetime.now().isoformat()
        }
        
        return self.api.store_document("structures", material)


class KnowledgeAPI:
    """
    知识库API
    
    整合所有组件，提供统一的知识库访问接口。
    """
    
    def __init__(self, config: APIConfig):
        self.config = config
        
        # 存储后端
        self.mongo = None
        self.postgres = None
        self.neo4j = None
        self.vector_store = None
        
        # 组件
        self.builder = None
        self.version_control = None
        self.search_engine = None
        self.embedding_provider = None
        
        # 工具
        self.pipeline = DataPipeline(self)
        self.query_builder = QueryBuilder(self)
        self.exporter = KnowledgeExporter(self)
        self.importer = KnowledgeImporter(self)
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化知识库"""
        try:
            # 初始化MongoDB
            from .storage.mongo_storage import MongoStorage, MongoConfig
            mongo_config = MongoConfig(
                host=self.config.mongodb_host,
                port=self.config.mongodb_port,
                database=self.config.mongodb_database
            )
            self.mongo = MongoStorage(mongo_config)
            self.mongo.connect()
            
            # 初始化PostgreSQL (可选)
            try:
                from .storage.postgres_storage import PostgresStorage, PostgresConfig
                pg_config = PostgresConfig(
                    host=self.config.postgres_host,
                    port=self.config.postgres_port,
                    database=self.config.postgres_database,
                    username=self.config.postgres_username,
                    password=self.config.postgres_password
                )
                self.postgres = PostgresStorage(pg_config)
                self.postgres.connect()
            except Exception as e:
                logger.warning(f"PostgreSQL not available: {e}")
            
            # 初始化Neo4j (可选)
            try:
                from .graph.neo4j_graph import Neo4jGraphDB, Neo4jConfig
                neo4j_config = Neo4jConfig(
                    uri=self.config.neo4j_uri,
                    username=self.config.neo4j_username,
                    password=self.config.neo4j_password
                )
                self.neo4j = Neo4jGraphDB(neo4j_config)
                self.neo4j.connect()
            except Exception as e:
                logger.warning(f"Neo4j not available: {e}")
            
            # 初始化向量存储
            from .embeddings.vector_store import (
                VectorConfig, create_vector_store, EmbeddingProvider
            )
            vector_config = VectorConfig(
                provider=self.config.vector_provider,
                dimension=self.config.vector_dimension,
                pinecone_api_key=self.config.pinecone_api_key,
                milvus_host=self.config.milvus_host,
                weaviate_url=self.config.weaviate_url
            )
            self.vector_store = create_vector_store(vector_config)
            self.vector_store.connect()
            
            # 初始化嵌入提供者
            self.embedding_provider = EmbeddingProvider()
            self.vector_store.set_embedding_provider(self.embedding_provider)
            
            # 初始化知识构建器
            from .knowledge_builder import KnowledgeBuilder
            self.builder = KnowledgeBuilder()
            
            # 初始化版本控制
            from .versioning.version_control import VersionControl
            self.version_control = VersionControl()
            
            # 初始化搜索引擎
            from .search.semantic_search import SemanticSearch, SearchConfig
            search_config = SearchConfig(
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider
            )
            self.search_engine = SemanticSearch(search_config)
            
            self._initialized = True
            logger.info("Knowledge API initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge API: {e}")
            return False
    
    def store_document(
        self,
        collection: str,
        data: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """
        存储文档
        
        Args:
            collection: 集合名称
            data: 数据
            tags: 标签
            
        Returns:
            记录ID
        """
        from .storage.base_storage import DataRecord
        
        record = DataRecord(
            data=data,
            tags=tags or [],
            source="api"
        )
        
        if self.mongo and self.mongo.is_connected:
            return self.mongo.insert(collection, record)
        elif self.postgres and self.postgres.is_connected:
            return self.postgres.insert(collection, record)
        else:
            raise RuntimeError("No storage backend available")
    
    def query_documents(
        self,
        collection: str,
        filters: Optional[List[Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        查询文档
        
        Args:
            collection: 集合名称
            filters: 过滤器
            limit: 限制数量
            
        Returns:
            结果列表
        """
        if self.mongo and self.mongo.is_connected:
            result = self.mongo.find(collection, filters=filters, page_size=limit)
            return [r.data for r in result.records]
        elif self.postgres and self.postgres.is_connected:
            result = self.postgres.find(collection, filters=filters, page_size=limit)
            return [r.data for r in result.records]
        else:
            return []
    
    def semantic_search(self, query: Any) -> List[Dict[str, Any]]:
        """
        语义搜索
        
        Args:
            query: 搜索查询
            
        Returns:
            结果列表
        """
        if self.search_engine:
            results = self.search_engine.search(query)
            return [r.to_dict() for r in results]
        return []
    
    def get_knowledge_graph(self) -> Dict[str, Any]:
        """获取知识图谱"""
        if self.builder:
            return self.builder.get_knowledge_graph()
        return {}
    
    def query_graph(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        执行图查询
        
        Args:
            cypher_query: Cypher查询语句
            
        Returns:
            查询结果
        """
        if self.neo4j and self.neo4j.is_connected:
            return self.neo4j.execute_query(cypher_query)
        return []
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "initialized": self._initialized,
            "mongo": self.mongo.is_connected if self.mongo else False,
            "postgres": self.postgres.is_connected if self.postgres else False,
            "neo4j": self.neo4j.is_connected if self.neo4j else False,
            "vector_store": self.vector_store.is_connected if self.vector_store else False
        }
    
    def close(self):
        """关闭连接"""
        if self.mongo:
            self.mongo.disconnect()
        if self.postgres:
            self.postgres.disconnect()
        if self.neo4j:
            self.neo4j.disconnect()
        if self.vector_store:
            self.vector_store.disconnect()
        
        logger.info("Knowledge API connections closed")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_knowledge_api(config: Optional[Union[APIConfig, Dict[str, Any]]] = None) -> KnowledgeAPI:
    """
    工厂函数：创建知识库API
    
    Args:
        config: 配置对象或字典
        
    Returns:
        知识库API实例
    """
    if config is None:
        config = APIConfig()
    elif isinstance(config, dict):
        config = APIConfig.from_dict(config)
    
    api = KnowledgeAPI(config)
    api.initialize()
    return api
