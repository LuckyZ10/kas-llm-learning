"""
Knowledge Base Module - 知识库模块
===============================
提供材料科学知识的持久化存储、知识图谱构建和语义搜索功能。

本模块整合了多种数据库技术：
- MongoDB: 文档存储，适合非结构化材料数据
- PostgreSQL: 关系型存储，适合结构化查询
- Neo4j: 图数据库，支持知识图谱推理
- 向量数据库(Pinecone/Milvus/Weaviate): 语义搜索

Author: DFT-LAMMPS Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DFT-LAMMPS Team"

# 存储层
from .storage.base_storage import (
    BaseStorage,
    StorageConfig,
    QueryFilter,
    QueryResult,
    DataRecord,
    StorageType
)
from .storage.mongo_storage import MongoStorage, MongoConfig
from .storage.postgres_storage import PostgresStorage, PostgresConfig

# 图数据库层
from .graph.neo4j_graph import (
    Neo4jGraphDB,
    Neo4jConfig,
    GraphQuery,
    GraphPattern,
    PathQuery,
    NodeSpec,
    RelationSpec,
    GraphPath,
    GraphMetrics
)

# 嵌入层
from .embeddings.vector_store import (
    VectorStore,
    VectorConfig,
    EmbeddingProvider,
    PineconeStore,
    MilvusStore,
    WeaviateStore,
    LocalVectorStore,
    SimilarityMetric,
    VectorSearchResult,
    create_vector_store
)

# 搜索层
from .search.semantic_search import (
    SemanticSearch,
    SearchQuery,
    SearchResult,
    SearchConfig,
    HybridSearch,
    SearchAggregator
)

# 版本控制层
from .versioning.version_control import (
    VersionControl,
    VersionTag,
    CalculationVersion,
    DiffResult,
    BranchManager,
    VersionComparator,
    create_version_control
)

# 知识构建器
from .knowledge_builder import (
    KnowledgeBuilder,
    KnowledgeSchema,
    EntityNormalizer,
    RelationExtractor,
    KnowledgeMerger,
    MaterialOntology
)

# 集成接口
from .knowledge_api import (
    KnowledgeAPI,
    APIConfig,
    DataPipeline,
    QueryBuilder,
    KnowledgeExporter,
    KnowledgeImporter
)

__all__ = [
    # 存储
    "BaseStorage",
    "StorageConfig",
    "QueryFilter",
    "QueryResult",
    "DataRecord",
    "StorageType",
    "MongoStorage",
    "MongoConfig",
    "PostgresStorage",
    "PostgresConfig",
    
    # 图数据库
    "Neo4jGraphDB",
    "Neo4jConfig",
    "GraphQuery",
    "GraphPattern",
    "PathQuery",
    "NodeSpec",
    "RelationSpec",
    "GraphPath",
    "GraphMetrics",
    
    # 向量存储
    "VectorStore",
    "VectorConfig",
    "EmbeddingProvider",
    "PineconeStore",
    "MilvusStore",
    "WeaviateStore",
    "LocalVectorStore",
    "SimilarityMetric",
    "VectorSearchResult",
    
    # 搜索
    "SemanticSearch",
    "SearchQuery",
    "SearchResult",
    "SearchConfig",
    "HybridSearch",
    "SearchAggregator",
    
    # 版本控制
    "VersionControl",
    "VersionTag",
    "CalculationVersion",
    "DiffResult",
    "BranchManager",
    "VersionComparator",
    "create_version_control",
    
    # 知识构建
    "KnowledgeBuilder",
    "KnowledgeSchema",
    "EntityNormalizer",
    "RelationExtractor",
    "KnowledgeMerger",
    "MaterialOntology",
    
    # API
    "KnowledgeAPI",
    "APIConfig",
    "DataPipeline",
    "QueryBuilder",
    "KnowledgeExporter",
    "KnowledgeImporter",
    
    # 工厂函数
    "create_knowledge_base",
    "create_vector_store",
]


def create_knowledge_base(config: dict) -> "KnowledgeAPI":
    """
    工厂函数：创建知识库实例
    
    Args:
        config: 配置字典
        
    Example:
        >>> kb = create_knowledge_base({
        ...     "mongodb": {"host": "localhost", "port": 27017},
        ...     "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j"},
        ...     "vector_store": {"provider": "pinecone", "api_key": "xxx"}
        ... })
    """
    from .knowledge_api import KnowledgeAPI, APIConfig
    api_config = APIConfig.from_dict(config)
    return KnowledgeAPI(api_config)


def demo():
    """知识库模块演示"""
    print("=" * 80)
    print("📚 DFT-LAMMPS 知识库模块 - 功能演示")
    print("=" * 80)
    
    print("\n1️⃣ 存储层演示")
    print("-" * 40)
    print("✅ MongoStorage - 文档存储")
    print("✅ PostgresStorage - 关系型存储")
    print("✅ BaseStorage - 统一接口")
    
    print("\n2️⃣ 图数据库演示")
    print("-" * 40)
    print("✅ Neo4jGraphDB - 知识图谱")
    print("✅ GraphQuery - Cypher查询")
    print("✅ PathQuery - 路径推理")
    
    print("\n3️⃣ 向量存储演示")
    print("-" * 40)
    print("✅ PineconeStore - 云端向量数据库")
    print("✅ MilvusStore - 开源向量数据库")
    print("✅ WeaviateStore - 语义向量数据库")
    print("✅ LocalVectorStore - 本地向量存储")
    
    print("\n4️⃣ 语义搜索演示")
    print("-" * 40)
    print("✅ SemanticSearch - 语义搜索")
    print("✅ HybridSearch - 混合搜索")
    print("✅ SearchAggregator - 结果聚合")
    
    print("\n5️⃣ 版本控制演示")
    print("-" * 40)
    print("✅ VersionControl - 版本管理")
    print("✅ BranchManager - 分支管理")
    print("✅ VersionComparator - 版本对比")
    
    print("\n6️⃣ 知识构建演示")
    print("-" * 40)
    print("✅ KnowledgeBuilder - 知识构建")
    print("✅ MaterialOntology - 材料本体")
    print("✅ EntityNormalizer - 实体标准化")
    
    print("\n" + "=" * 80)
    print("✅ 知识库模块演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    demo()
