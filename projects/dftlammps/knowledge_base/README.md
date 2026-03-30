# Knowledge Base Module - 知识库模块

DFT-LAMMPS平台的数据库与知识图谱集成模块，提供材料科学数据的持久化存储、知识图谱构建和语义搜索功能。

## 功能特性

### 1. 多存储后端支持 (Storage Layer)
- **MongoDB**: 文档存储，适合非结构化材料数据
- **PostgreSQL**: 关系型存储，支持复杂查询和ACID事务
- 统一的存储接口，便于扩展

### 2. 图数据库集成 (Graph Layer)
- **Neo4j**: 知识图谱存储和查询
- 支持Cypher查询语言
- 图遍历和路径搜索
- 图算法集成 (PageRank, 社区检测, 最短路径等)

### 3. 向量数据库支持 (Embedding Layer)
- **Pinecone**: 云端向量数据库
- **Milvus**: 开源向量数据库
- **Weaviate**: 语义向量数据库
- **本地存储**: FAISS/numpy实现

### 4. 语义搜索 (Search Layer)
- 向量相似度搜索
- 混合搜索 (语义+关键词)
- 结果重排序和多样性优化

### 5. 版本控制 (Versioning Layer)
- 类似Git的版本管理
- 支持分支和标签
- 版本差异比较
- 计算结果历史追踪

### 6. 知识构建 (Knowledge Builder)
- 材料科学本体定义
- 实体标准化
- 关系自动抽取
- 知识图谱构建

## 快速开始

### 安装依赖

```bash
pip install pymongo psycopg2-binary neo4j pinecone-client pymilvus weaviate-client
```

### 基本使用

```python
from dftlammps.knowledge_base import create_knowledge_api, APIConfig

# 创建配置
config = APIConfig(
    mongodb_host="localhost",
    mongodb_port=27017,
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    vector_provider="local"
)

# 初始化知识库
with create_knowledge_api(config) as kb:
    # 存储材料数据
    material = {
        "formula": "Li3PS4",
        "name": "Lithium Thiophosphate",
        "properties": {"band_gap": 2.5}
    }
    record_id = kb.store_document("materials", material)
    
    # 语义搜索
    results = kb.query_builder.search(
        query="solid electrolyte materials",
        mode="hybrid",
        top_k=10
    )
```

### 知识图谱构建

```python
from dftlammps.knowledge_base import KnowledgeBuilder

builder = KnowledgeBuilder()

# 添加实体
material = builder.add_entity(
    entity_type="Material",
    name="Li3PS4",
    properties={"formula": "Li3PS4", "band_gap": 2.5}
)

# 添加关系
builder.add_relation(
    relation_type="HAS_PROPERTY",
    from_entity=material["id"],
    to_entity=property_id
)

# 导出到Neo4j
builder.export_to_neo4j(neo4j_db)
```

### 版本控制

```python
from dftlammps.knowledge_base import create_version_control

vc = create_version_control()

# 提交版本
v1 = vc.commit(
    data=calculation_result,
    message="Initial DFT calculation",
    author="researcher1"
)

# 打标签
vc.tag(v1.id, "v1.0", "Stable version")

# 比较版本
diff = vc.diff(v1.id, v2.id)
print(diff.summary())
```

## 模块结构

```
knowledge_base/
├── __init__.py              # 主模块接口
├── knowledge_api.py         # 统一API接口
├── knowledge_builder.py     # 知识图谱构建
├── storage/                 # 存储层
│   ├── base_storage.py      # 存储基类
│   ├── mongo_storage.py     # MongoDB实现
│   └── postgres_storage.py  # PostgreSQL实现
├── graph/                   # 图数据库层
│   └── neo4j_graph.py       # Neo4j实现
├── embeddings/              # 向量存储层
│   └── vector_store.py      # 向量存储实现
├── search/                  # 搜索层
│   └── semantic_search.py   # 语义搜索
├── versioning/              # 版本控制层
│   └── version_control.py   # 版本控制实现
├── tests/                   # 测试
│   └── test_knowledge_base.py
└── examples/                # 示例
    └── basic_usage.py
```

## 配置选项

### MongoDB配置
```python
MongoConfig(
    host="localhost",
    port=27017,
    database="dftlammps_kb",
    max_pool_size=100
)
```

### PostgreSQL配置
```python
PostgresConfig(
    host="localhost",
    port=5432,
    database="dftlammps_kb",
    username="postgres",
    password="password"
)
```

### Neo4j配置
```python
Neo4jConfig(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password"
)
```

### 向量存储配置
```python
VectorConfig(
    provider="pinecone",  # pinecone, milvus, weaviate, local
    dimension=1536,
    pinecone_api_key="your-api-key"
)
```

## API参考

### KnowledgeAPI

主API类，整合所有功能：

- `store_document(collection, data)` - 存储文档
- `query_documents(collection, filters)` - 查询文档
- `semantic_search(query)` - 语义搜索
- `query_graph(cypher)` - 图查询
- `pipeline.ingest_calculation(data)` - 摄入计算结果

### KnowledgeBuilder

知识图谱构建器：

- `add_entity(type, name, properties)` - 添加实体
- `add_relation(type, from, to)` - 添加关系
- `build_from_calculation(data)` - 从计算结果构建
- `export_to_neo4j(db)` - 导出到Neo4j

### VersionControl

版本控制系统：

- `commit(data, message)` - 提交版本
- `checkout(version_id)` - 检出版本
- `diff(v1, v2)` - 比较版本
- `tag(version_id, name)` - 打标签

## 示例

运行示例：

```bash
python -m dftlammps.knowledge_base.examples.basic_usage
```

运行测试：

```bash
python -m dftlammps.knowledge_base.tests.test_knowledge_base
```

## 性能优化

1. **索引**: 为常用查询字段创建索引
2. **批处理**: 使用批量插入减少IO
3. **缓存**: 启用搜索结果缓存
4. **连接池**: 配置合适的连接池大小

## 故障排除

### 连接问题
- 检查数据库服务是否运行
- 验证连接配置
- 检查防火墙设置

### 性能问题
- 启用查询日志分析
- 检查索引使用情况
- 考虑数据分片

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
