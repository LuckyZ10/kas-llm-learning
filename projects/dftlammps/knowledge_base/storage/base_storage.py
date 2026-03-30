"""
Base Storage Interface - 存储基类
===============================
定义统一的数据存储接口，支持多种后端实现。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Iterator, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import json
import hashlib
from pathlib import Path


class StorageType(Enum):
    """存储类型枚举"""
    DOCUMENT = auto()      # 文档存储 (MongoDB)
    RELATIONAL = auto()    # 关系型存储 (PostgreSQL)
    GRAPH = auto()         # 图存储 (Neo4j)
    VECTOR = auto()        # 向量存储
    KEY_VALUE = auto()     # 键值存储
    TIME_SERIES = auto()   # 时序存储


@dataclass
class StorageConfig:
    """存储配置基类"""
    storage_type: StorageType
    host: str = "localhost"
    port: int = 0
    database: str = "dftlammps"
    username: Optional[str] = None
    password: Optional[str] = None
    ssl: bool = False
    timeout: int = 30
    max_pool_size: int = 10
    retry_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage_type": self.storage_type.name,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "ssl": self.ssl,
            "timeout": self.timeout,
            "max_pool_size": self.max_pool_size
        }


@dataclass
class QueryFilter:
    """查询过滤器"""
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, nin, regex, exists
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value
        }


@dataclass
class DataRecord:
    """数据记录"""
    id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: int = 1
    checksum: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """计算数据校验和"""
        data_str = json.dumps(self.data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "data": self.data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "version": self.version,
            "checksum": self.checksum,
            "tags": self.tags,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataRecord":
        """从字典创建记录"""
        return cls(
            id=data.get("id"),
            data=data.get("data", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            version=data.get("version", 1),
            checksum=data.get("checksum"),
            tags=data.get("tags", []),
            source=data.get("source")
        )


@dataclass
class QueryResult:
    """查询结果"""
    records: List[DataRecord] = field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 100
    execution_time_ms: float = 0.0
    query_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __iter__(self):
        return iter(self.records)
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, index) -> DataRecord:
        return self.records[index]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "total_count": self.total_count,
            "page": self.page,
            "page_size": self.page_size,
            "execution_time_ms": self.execution_time_ms,
            "query_metadata": self.query_metadata
        }


class BaseStorage(ABC):
    """
    存储基类 - 定义统一接口
    
    所有存储后端(MongoDB, PostgreSQL等)都需要实现这个接口
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self._connected = False
        self._connection = None
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接到存储后端
        
        Returns:
            连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开连接
        
        Returns:
            断开是否成功
        """
        pass
    
    @abstractmethod
    def create_collection(self, name: str, schema: Optional[Dict] = None) -> bool:
        """
        创建集合/表
        
        Args:
            name: 集合名称
            schema: 可选的结构定义
            
        Returns:
            创建是否成功
        """
        pass
    
    @abstractmethod
    def drop_collection(self, name: str) -> bool:
        """
        删除集合/表
        
        Args:
            name: 集合名称
            
        Returns:
            删除是否成功
        """
        pass
    
    @abstractmethod
    def insert(self, collection: str, record: DataRecord) -> str:
        """
        插入记录
        
        Args:
            collection: 集合名称
            record: 数据记录
            
        Returns:
            记录ID
        """
        pass
    
    @abstractmethod
    def insert_many(self, collection: str, records: List[DataRecord]) -> List[str]:
        """
        批量插入记录
        
        Args:
            collection: 集合名称
            records: 数据记录列表
            
        Returns:
            记录ID列表
        """
        pass
    
    @abstractmethod
    def find_by_id(self, collection: str, record_id: str) -> Optional[DataRecord]:
        """
        根据ID查找记录
        
        Args:
            collection: 集合名称
            record_id: 记录ID
            
        Returns:
            数据记录或None
        """
        pass
    
    @abstractmethod
    def find(
        self,
        collection: str,
        filters: Optional[List[QueryFilter]] = None,
        sort: Optional[Dict[str, str]] = None,
        page: int = 1,
        page_size: int = 100
    ) -> QueryResult:
        """
        查询记录
        
        Args:
            collection: 集合名称
            filters: 过滤器列表
            sort: 排序规则 {"field": "asc|desc"}
            page: 页码
            page_size: 每页大小
            
        Returns:
            查询结果
        """
        pass
    
    @abstractmethod
    def update(self, collection: str, record_id: str, data: Dict[str, Any]) -> bool:
        """
        更新记录
        
        Args:
            collection: 集合名称
            record_id: 记录ID
            data: 更新数据
            
        Returns:
            更新是否成功
        """
        pass
    
    @abstractmethod
    def update_many(
        self,
        collection: str,
        filters: List[QueryFilter],
        data: Dict[str, Any]
    ) -> int:
        """
        批量更新记录
        
        Args:
            collection: 集合名称
            filters: 过滤器列表
            data: 更新数据
            
        Returns:
            更新的记录数
        """
        pass
    
    @abstractmethod
    def delete(self, collection: str, record_id: str) -> bool:
        """
        删除记录
        
        Args:
            collection: 集合名称
            record_id: 记录ID
            
        Returns:
            删除是否成功
        """
        pass
    
    @abstractmethod
    def delete_many(self, collection: str, filters: List[QueryFilter]) -> int:
        """
        批量删除记录
        
        Args:
            collection: 集合名称
            filters: 过滤器列表
            
        Returns:
            删除的记录数
        """
        pass
    
    @abstractmethod
    def count(self, collection: str, filters: Optional[List[QueryFilter]] = None) -> int:
        """
        计数记录
        
        Args:
            collection: 集合名称
            filters: 过滤器列表
            
        Returns:
            记录数
        """
        pass
    
    @abstractmethod
    def aggregate(
        self,
        collection: str,
        pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        聚合查询
        
        Args:
            collection: 集合名称
            pipeline: 聚合管道
            
        Returns:
            聚合结果
        """
        pass
    
    @abstractmethod
    def create_index(
        self,
        collection: str,
        fields: List[str],
        index_type: str = "btree",
        unique: bool = False
    ) -> bool:
        """
        创建索引
        
        Args:
            collection: 集合名称
            fields: 字段列表
            index_type: 索引类型 (btree, hash, text, vector)
            unique: 是否唯一
            
        Returns:
            创建是否成功
        """
        pass
    
    @abstractmethod
    def transaction(self) -> "StorageTransaction":
        """
        开始事务
        
        Returns:
            事务上下文管理器
        """
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        return {
            "connected": self._connected,
            "storage_type": self.config.storage_type.name,
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database
        }
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class StorageTransaction:
    """存储事务上下文管理器"""
    
    def __init__(self, storage: BaseStorage):
        self.storage = storage
        self._active = False
    
    def __enter__(self):
        self.begin()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
    
    def begin(self):
        """开始事务"""
        self._active = True
    
    def commit(self):
        """提交事务"""
        self._active = False
    
    def rollback(self):
        """回滚事务"""
        self._active = False
    
    @property
    def is_active(self) -> bool:
        return self._active


class MultiStorageManager:
    """多存储管理器 - 统一管理多个存储后端"""
    
    def __init__(self):
        self._storages: Dict[str, BaseStorage] = {}
        self._default_storage: Optional[str] = None
    
    def register(
        self,
        name: str,
        storage: BaseStorage,
        default: bool = False
    ):
        """
        注册存储后端
        
        Args:
            name: 存储名称
            storage: 存储实例
            default: 是否设为默认
        """
        self._storages[name] = storage
        if default or self._default_storage is None:
            self._default_storage = name
    
    def get(self, name: Optional[str] = None) -> BaseStorage:
        """
        获取存储实例
        
        Args:
            name: 存储名称，None则返回默认
            
        Returns:
            存储实例
        """
        if name is None:
            name = self._default_storage
        if name not in self._storages:
            raise ValueError(f"Storage '{name}' not registered")
        return self._storages[name]
    
    def list_storages(self) -> List[str]:
        """列出所有存储"""
        return list(self._storages.keys())
    
    def connect_all(self) -> Dict[str, bool]:
        """连接所有存储"""
        results = {}
        for name, storage in self._storages.items():
            try:
                results[name] = storage.connect()
            except Exception as e:
                results[name] = False
        return results
    
    def disconnect_all(self) -> Dict[str, bool]:
        """断开所有存储"""
        results = {}
        for name, storage in self._storages.items():
            try:
                results[name] = storage.disconnect()
            except Exception as e:
                results[name] = False
        return results
    
    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """检查所有存储健康状态"""
        return {name: storage.health_check() for name, storage in self._storages.items()}


def create_filter(field: str, operator: str, value: Any) -> QueryFilter:
    """创建查询过滤器的便捷函数"""
    return QueryFilter(field=field, operator=operator, value=value)


def create_record(data: Dict[str, Any], **kwargs) -> DataRecord:
    """创建数据记录的便捷函数"""
    return DataRecord(data=data, **kwargs)
