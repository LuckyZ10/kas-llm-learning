"""
Vector Store - 向量数据库存储
===========================
支持多种向量数据库：Pinecone, Milvus, Weaviate
用于语义搜索和相似度查询。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """相似度度量"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class VectorConfig:
    """向量存储配置"""
    provider: str = "local"  # pinecone, milvus, weaviate, local
    collection_name: str = "materials"
    dimension: int = 1536  # 默认embedding维度
    metric: SimilarityMetric = SimilarityMetric.COSINE
    
    # Pinecone配置
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp"
    
    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: Optional[str] = None
    milvus_password: Optional[str] = None
    
    # Weaviate配置
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: Optional[str] = None
    
    # 本地存储配置
    local_path: str = "./vector_store"
    
    # 通用配置
    batch_size: int = 100
    max_retries: int = 3


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    id: str
    score: float
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "vector": self.vector,
            "metadata": self.metadata
        }


class EmbeddingProvider:
    """
    嵌入向量提供者
    
    支持多种embedding模型
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        local_model_path: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.local_model_path = local_model_path
        self._model = None
    
    def embed_text(self, text: str) -> List[float]:
        """文本向量化"""
        if "openai" in self.model_name or "ada" in self.model_name:
            return self._embed_openai(text)
        elif "sentence-transformers" in self.model_name:
            return self._embed_sentence_transformers(text)
        else:
            # 默认使用简单的哈希embedding (仅用于测试)
            return self._embed_hash(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化"""
        return [self.embed_text(t) for t in texts]
    
    def embed_material(self, material: Dict[str, Any]) -> List[float]:
        """材料向量化"""
        # 构建材料描述文本
        description = self._material_to_text(material)
        return self.embed_text(description)
    
    def _material_to_text(self, material: Dict[str, Any]) -> str:
        """将材料数据转换为文本描述"""
        parts = []
        
        if "formula" in material:
            parts.append(f"Chemical formula: {material['formula']}")
        if "structure" in material:
            parts.append(f"Crystal structure: {material['structure']}")
        if "properties" in material:
            props = material["properties"]
            if "band_gap" in props:
                parts.append(f"Band gap: {props['band_gap']} eV")
            if "formation_energy" in props:
                parts.append(f"Formation energy: {props['formation_energy']} eV/atom")
        
        return " ".join(parts) if parts else str(material)
    
    def _embed_openai(self, text: str) -> List[float]:
        """使用OpenAI API向量化"""
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.Embedding.create(
                model=self.model_name,
                input=text
            )
            return response["data"][0]["embedding"]
        except ImportError:
            logger.error("openai not installed")
            return self._embed_hash(text)
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return self._embed_hash(text)
    
    def _embed_sentence_transformers(self, text: str) -> List[float]:
        """使用Sentence Transformers向量化"""
        try:
            from sentence_transformers import SentenceTransformer
            
            if self._model is None:
                self._model = SentenceTransformer(self.local_model_path or "all-MiniLM-L6-v2")
            
            embedding = self._model.encode(text)
            return embedding.tolist()
        except ImportError:
            logger.error("sentence_transformers not installed")
            return self._embed_hash(text)
        except Exception as e:
            logger.error(f"Sentence transformers embedding failed: {e}")
            return self._embed_hash(text)
    
    def _embed_hash(self, text: str, dimension: int = 1536) -> List[float]:
        """使用哈希函数生成伪embedding (仅用于测试)"""
        import hashlib
        
        # 使用多个哈希来生成向量
        np.random.seed(int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32))
        vector = np.random.randn(dimension).astype(np.float32)
        # 归一化
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()


class VectorStore(ABC):
    """向量存储抽象基类"""
    
    def __init__(self, config: VectorConfig):
        self.config = config
        self._connected = False
        self.embedding_provider: Optional[EmbeddingProvider] = None
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def set_embedding_provider(self, provider: EmbeddingProvider):
        """设置嵌入提供者"""
        self.embedding_provider = provider
    
    @abstractmethod
    def connect(self) -> bool:
        """连接到向量数据库"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> bool:
        """创建集合/索引"""
        pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """删除集合"""
        pass
    
    @abstractmethod
    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """插入或更新向量"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """向量相似度搜索"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """删除向量"""
        pass
    
    @abstractmethod
    def get(self, ids: List[str]) -> List[VectorSearchResult]:
        """获取向量"""
        pass
    
    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """文本相似度搜索"""
        if self.embedding_provider is None:
            raise ValueError("Embedding provider not set")
        
        query_vector = self.embedding_provider.embed_text(query_text)
        return self.search(query_vector, top_k, filter)
    
    def upsert_texts(
        self,
        texts: List[str],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """插入文本 (自动向量化)"""
        if self.embedding_provider is None:
            raise ValueError("Embedding provider not set")
        
        vectors = self.embedding_provider.embed_texts(texts)
        return self.upsert(vectors, ids, metadata)
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class PineconeStore(VectorStore):
    """Pinecone向量存储"""
    
    def __init__(self, config: Optional[VectorConfig] = None):
        if config is None:
            config = VectorConfig(provider="pinecone")
        super().__init__(config)
        self._index = None
        self._pinecone = None
    
    def connect(self) -> bool:
        """连接到Pinecone"""
        try:
            import pinecone
            
            pinecone.init(
                api_key=self.config.pinecone_api_key,
                environment=self.config.pinecone_environment
            )
            
            self._pinecone = pinecone
            self._connected = True
            
            logger.info("Connected to Pinecone")
            return True
            
        except ImportError:
            logger.error("pinecone-client not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        # Pinecone不需要显式断开
        self._connected = False
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> bool:
        """创建索引"""
        try:
            if name not in self._pinecone.list_indexes():
                self._pinecone.create_index(
                    name=name,
                    dimension=dimension,
                    metric=metric.value
                )
                logger.info(f"Created Pinecone index: {name}")
            
            self._index = self._pinecone.Index(name)
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        """删除索引"""
        try:
            self._pinecone.delete_index(name)
            logger.info(f"Deleted Pinecone index: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Pinecone index: {e}")
            return False
    
    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """插入向量"""
        try:
            if self._index is None:
                self._index = self._pinecone.Index(self.config.collection_name)
            
            # 构建upsert数据
            vectors_data = []
            for i, (vector, id_) in enumerate(zip(vectors, ids)):
                meta = metadata[i] if metadata and i < len(metadata) else {}
                # 清理metadata中的非字符串值
                meta = self._clean_metadata(meta)
                vectors_data.append((id_, vector, meta))
            
            # 批量upsert
            self._index.upsert(vectors=vectors_data, batch_size=self.config.batch_size)
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            return False
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """搜索相似向量"""
        try:
            if self._index is None:
                self._index = self._pinecone.Index(self.config.collection_name)
            
            results = self._index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter,
                include_metadata=True
            )
            
            return [
                VectorSearchResult(
                    id=match["id"],
                    score=match["score"],
                    metadata=match.get("metadata", {})
                )
                for match in results["matches"]
            ]
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def delete(self, ids: List[str]) -> bool:
        """删除向量"""
        try:
            if self._index is None:
                self._index = self._pinecone.Index(self.config.collection_name)
            
            self._index.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    def get(self, ids: List[str]) -> List[VectorSearchResult]:
        """获取向量"""
        try:
            if self._index is None:
                self._index = self._pinecone.Index(self.config.collection_name)
            
            results = self._index.fetch(ids=ids)
            
            return [
                VectorSearchResult(
                    id=id_,
                    score=1.0,
                    vector=vec.get("values", []),
                    metadata=vec.get("metadata", {})
                )
                for id_, vec in results["vectors"].items()
            ]
            
        except Exception as e:
            logger.error(f"Failed to get vectors: {e}")
            return []
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清理metadata，只保留支持的类型"""
        cleaned = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool, list)):
                # Pinecone只支持基本类型和字符串列表
                if isinstance(v, list):
                    cleaned[k] = [str(x) for x in v]
                else:
                    cleaned[k] = v
            else:
                cleaned[k] = str(v)
        return cleaned


class MilvusStore(VectorStore):
    """Milvus向量存储"""
    
    def __init__(self, config: Optional[VectorConfig] = None):
        if config is None:
            config = VectorConfig(provider="milvus")
        super().__init__(config)
        self._collection = None
        self._connections = None
    
    def connect(self) -> bool:
        """连接到Milvus"""
        try:
            from pymilvus import connections
            
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                user=self.config.milvus_user or "",
                password=self.config.milvus_password or ""
            )
            
            self._connections = connections
            self._connected = True
            
            logger.info(f"Connected to Milvus at {self.config.milvus_host}:{self.config.milvus_port}")
            return True
            
        except ImportError:
            logger.error("pymilvus not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self._connections:
            self._connections.disconnect("default")
        self._connected = False
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> bool:
        """创建集合"""
        try:
            from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
            
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(fields, description=f"Collection for {name}")
            
            self._collection = Collection(name=name, schema=schema)
            
            # 创建索引
            index_params = {
                "metric_type": metric.value.upper(),
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self._collection.create_index(field_name="vector", index_params=index_params)
            
            logger.info(f"Created Milvus collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Milvus collection: {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        """删除集合"""
        try:
            from pymilvus import utility
            utility.drop_collection(name)
            logger.info(f"Deleted Milvus collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Milvus collection: {e}")
            return False
    
    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """插入向量"""
        try:
            from pymilvus import Collection
            
            if self._collection is None:
                self._collection = Collection(self.config.collection_name)
            
            # 准备数据
            entities = [
                ids,
                vectors,
                metadata if metadata else [{} for _ in ids]
            ]
            
            self._collection.insert(entities)
            self._collection.flush()
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            return False
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """搜索相似向量"""
        try:
            from pymilvus import Collection
            
            if self._collection is None:
                self._collection = Collection(self.config.collection_name)
            
            self._collection.load()
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = self._collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=self._build_filter(filter) if filter else None,
                output_fields=["metadata"]
            )
            
            return [
                VectorSearchResult(
                    id=hit.id,
                    score=hit.distance,
                    metadata=hit.entity.get("metadata", {})
                )
                for hit in results[0]
            ]
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def delete(self, ids: List[str]) -> bool:
        """删除向量"""
        try:
            expr = f'id in {ids}'
            self._collection.delete(expr)
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    def get(self, ids: List[str]) -> List[VectorSearchResult]:
        """获取向量"""
        try:
            expr = f'id in {ids}'
            results = self._collection.query(
                expr=expr,
                output_fields=["vector", "metadata"]
            )
            
            return [
                VectorSearchResult(
                    id=r["id"],
                    score=1.0,
                    vector=r["vector"],
                    metadata=r.get("metadata", {})
                )
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get vectors: {e}")
            return []
    
    def _build_filter(self, filter_dict: Dict[str, Any]) -> str:
        """构建Milvus过滤表达式"""
        conditions = []
        for k, v in filter_dict.items():
            if isinstance(v, str):
                conditions.append(f'{k} == "{v}"')
            else:
                conditions.append(f'{k} == {v}')
        return " and ".join(conditions)


class WeaviateStore(VectorStore):
    """Weaviate向量存储"""
    
    def __init__(self, config: Optional[VectorConfig] = None):
        if config is None:
            config = VectorConfig(provider="weaviate")
        super().__init__(config)
        self._client = None
    
    def connect(self) -> bool:
        """连接到Weaviate"""
        try:
            import weaviate
            
            auth_config = None
            if self.config.weaviate_api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.config.weaviate_api_key)
            
            self._client = weaviate.Client(
                url=self.config.weaviate_url,
                auth_client_secret=auth_config
            )
            
            # 验证连接
            self._client.schema.get()
            self._connected = True
            
            logger.info(f"Connected to Weaviate at {self.config.weaviate_url}")
            return True
            
        except ImportError:
            logger.error("weaviate-client not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self._client = None
        self._connected = False
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> bool:
        """创建Class"""
        try:
            class_obj = {
                "class": name,
                "vectorizer": "none",  # 我们手动提供向量
                "moduleConfig": {
                    "text2vec-transformers": {"skip": True}
                },
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "metadata", "dataType": ["text"]}  # JSON as string
                ],
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": metric.value,
                    "ef": 256,
                    "efConstruction": 128,
                    "maxConnections": 64
                }
            }
            
            self._client.schema.create_class(class_obj)
            logger.info(f"Created Weaviate class: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Weaviate class: {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        """删除Class"""
        try:
            self._client.schema.delete_class(name)
            logger.info(f"Deleted Weaviate class: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Weaviate class: {e}")
            return False
    
    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """插入向量"""
        try:
            from weaviate.util import generate_uuid5
            
            with self._client.batch as batch:
                batch.batch_size = self.config.batch_size
                
                for i, (vector, id_) in enumerate(zip(vectors, ids)):
                    meta = metadata[i] if metadata and i < len(metadata) else {}
                    
                    data_object = {
                        "content": meta.get("content", ""),
                        "metadata": json.dumps(meta)
                    }
                    
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=self.config.collection_name,
                        vector=vector,
                        uuid=generate_uuid5(id_)
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            return False
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """搜索相似向量"""
        try:
            near_vector = {"vector": query_vector}
            
            where_filter = None
            if filter:
                # 构建where filter
                where_filter = self._build_where_filter(filter)
            
            results = (
                self._client.query
                .get(self.config.collection_name, ["content", "metadata"])
                .with_near_vector(near_vector)
                .with_limit(top_k)
                .with_additional(["distance", "id"])
            )
            
            if where_filter:
                results = results.with_where(where_filter)
            
            response = results.do()
            
            return [
                VectorSearchResult(
                    id=obj["_additional"]["id"],
                    score=1 - obj["_additional"]["distance"],  # 转换为相似度
                    metadata=json.loads(obj.get("metadata", "{}"))
                )
                for obj in response["data"]["Get"][self.config.collection_name]
            ]
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def delete(self, ids: List[str]) -> bool:
        """删除向量"""
        try:
            for id_ in ids:
                self._client.data_object.delete(id_, self.config.collection_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    def get(self, ids: List[str]) -> List[VectorSearchResult]:
        """获取向量"""
        try:
            results = []
            for id_ in ids:
                obj = self._client.data_object.get_by_id(
                    id_,
                    class_name=self.config.collection_name,
                    additional_properties=["vector"]
                )
                if obj:
                    results.append(VectorSearchResult(
                        id=id_,
                        score=1.0,
                        vector=obj.get("vector"),
                        metadata=json.loads(obj.get("properties", {}).get("metadata", "{}"))
                    ))
            return results
        except Exception as e:
            logger.error(f"Failed to get vectors: {e}")
            return []
    
    def _build_where_filter(self, filter_dict: Dict[str, Any]) -> Dict:
        """构建Weaviate where filter"""
        # 简化的filter构建
        operands = []
        for k, v in filter_dict.items():
            operands.append({
                "path": [k],
                "operator": "Equal",
                "valueText": str(v)
            })
        
        if len(operands) == 1:
            return operands[0]
        else:
            return {"operator": "And", "operands": operands}


class LocalVectorStore(VectorStore):
    """
    本地向量存储
    
    使用FAISS或简单numpy实现，适合开发和测试
    """
    
    def __init__(self, config: Optional[VectorConfig] = None):
        if config is None:
            config = VectorConfig(provider="local")
        super().__init__(config)
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._use_faiss = False
        self._faiss_index = None
    
    def connect(self) -> bool:
        """连接到本地存储"""
        try:
            # 尝试导入FAISS
            import faiss
            self._use_faiss = True
            logger.info("Using FAISS for local vector store")
        except ImportError:
            logger.info("FAISS not available, using numpy implementation")
            self._use_faiss = False
        
        self._connected = True
        return True
    
    def disconnect(self):
        """断开连接"""
        self._connected = False
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> bool:
        """创建集合"""
        if self._use_faiss:
            import faiss
            
            if metric == SimilarityMetric.COSINE or metric == SimilarityMetric.DOT_PRODUCT:
                self._faiss_index = faiss.IndexFlatIP(dimension)  # 内积
            else:
                self._faiss_index = faiss.IndexFlatL2(dimension)  # L2距离
        
        self._vectors = {}
        self._metadata = {}
        return True
    
    def delete_collection(self, name: str) -> bool:
        """删除集合"""
        self._vectors = {}
        self._metadata = {}
        self._faiss_index = None
        return True
    
    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """插入向量"""
        import numpy as np
        
        for i, (vector, id_) in enumerate(zip(vectors, ids)):
            self._vectors[id_] = vector
            if metadata and i < len(metadata):
                self._metadata[id_] = metadata[i]
        
        # 更新FAISS索引
        if self._use_faiss and self._faiss_index is not None:
            vectors_array = np.array(list(self._vectors.values())).astype("float32")
            self._faiss_index.reset()
            self._faiss_index.add(vectors_array)
        
        return True
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """搜索相似向量"""
        import numpy as np
        
        query = np.array(query_vector).astype("float32")
        
        if self._use_faiss and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query.reshape(1, -1), top_k)
            
            results = []
            id_list = list(self._vectors.keys())
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0 and idx < len(id_list):
                    id_ = id_list[idx]
                    # FAISS返回的是距离，需要转换
                    sim_score = float(score)
                    if self._faiss_index.metric_type == faiss.METRIC_L2:
                        sim_score = 1.0 / (1.0 + sim_score)  # L2转相似度
                    
                    results.append(VectorSearchResult(
                        id=id_,
                        score=sim_score,
                        metadata=self._metadata.get(id_, {})
                    ))
            return results
        else:
            # 使用numpy计算相似度
            results = []
            for id_, vector in self._vectors.items():
                vec = np.array(vector)
                similarity = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
                results.append((id_, float(similarity)))
            
            # 排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            return [
                VectorSearchResult(
                    id=id_,
                    score=score,
                    metadata=self._metadata.get(id_, {})
                )
                for id_, score in results[:top_k]
            ]
    
    def delete(self, ids: List[str]) -> bool:
        """删除向量"""
        for id_ in ids:
            self._vectors.pop(id_, None)
            self._metadata.pop(id_, None)
        
        # 重建FAISS索引
        if self._use_faiss and self._vectors:
            import numpy as np
            vectors_array = np.array(list(self._vectors.values())).astype("float32")
            self._faiss_index.reset()
            self._faiss_index.add(vectors_array)
        
        return True
    
    def get(self, ids: List[str]) -> List[VectorSearchResult]:
        """获取向量"""
        results = []
        for id_ in ids:
            if id_ in self._vectors:
                results.append(VectorSearchResult(
                    id=id_,
                    score=1.0,
                    vector=self._vectors[id_],
                    metadata=self._metadata.get(id_, {})
                ))
        return results
    
    def save_to_disk(self, path: str):
        """保存到磁盘"""
        import pickle
        data = {
            "vectors": self._vectors,
            "metadata": self._metadata
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved vector store to {path}")
    
    def load_from_disk(self, path: str):
        """从磁盘加载"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._vectors = data["vectors"]
        self._metadata = data["metadata"]
        
        # 重建FAISS索引
        if self._use_faiss and self._vectors:
            import numpy as np
            vectors_array = np.array(list(self._vectors.values())).astype("float32")
            if self._faiss_index:
                self._faiss_index.reset()
                self._faiss_index.add(vectors_array)
        
        logger.info(f"Loaded vector store from {path}")


def create_vector_store(config: VectorConfig) -> VectorStore:
    """
    工厂函数：创建向量存储实例
    
    Args:
        config: 向量存储配置
        
    Returns:
        向量存储实例
    """
    if config.provider == "pinecone":
        return PineconeStore(config)
    elif config.provider == "milvus":
        return MilvusStore(config)
    elif config.provider == "weaviate":
        return WeaviateStore(config)
    else:
        return LocalVectorStore(config)
