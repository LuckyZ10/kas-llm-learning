"""
Storage Module - 存储模块
=======================
提供多种存储后端支持。
"""

from .base_storage import (
    BaseStorage,
    StorageConfig,
    StorageType,
    QueryFilter,
    DataRecord,
    QueryResult,
    StorageTransaction,
    MultiStorageManager,
    create_filter,
    create_record
)

from .mongo_storage import MongoStorage, MongoConfig, MongoTransaction
from .postgres_storage import PostgresStorage, PostgresConfig, PostgresTransaction

__all__ = [
    "BaseStorage",
    "StorageConfig",
    "StorageType",
    "QueryFilter",
    "DataRecord",
    "QueryResult",
    "StorageTransaction",
    "MultiStorageManager",
    "create_filter",
    "create_record",
    "MongoStorage",
    "MongoConfig",
    "MongoTransaction",
    "PostgresStorage",
    "PostgresConfig",
    "PostgresTransaction",
]
