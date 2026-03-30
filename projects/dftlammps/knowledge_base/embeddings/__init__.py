"""
Embeddings Module - 嵌入模块
===========================
提供向量存储支持。
"""

from .vector_store import (
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

__all__ = [
    "VectorStore",
    "VectorConfig",
    "EmbeddingProvider",
    "PineconeStore",
    "MilvusStore",
    "WeaviateStore",
    "LocalVectorStore",
    "SimilarityMetric",
    "VectorSearchResult",
    "create_vector_store",
]
