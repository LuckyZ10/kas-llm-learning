"""
MongoDB Storage - MongoDB存储实现
================================
基于pymongo的文档存储实现，适合非结构化材料数据。
"""

from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass
from datetime import datetime
import logging

from .base_storage import (
    BaseStorage, StorageConfig, StorageType, 
    QueryFilter, DataRecord, QueryResult, StorageTransaction
)

logger = logging.getLogger(__name__)


@dataclass
class MongoConfig(StorageConfig):
    """MongoDB配置"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "dftlammps",
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_source: str = "admin",
        replica_set: Optional[str] = None,
        ssl: bool = False,
        max_pool_size: int = 100,
        min_pool_size: int = 10,
        max_idle_time_ms: int = 60000,
        wait_queue_timeout_ms: int = 5000,
        server_selection_timeout_ms: int = 30000,
        **kwargs
    ):
        super().__init__(
            storage_type=StorageType.DOCUMENT,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            ssl=ssl,
            max_pool_size=max_pool_size
        )
        self.auth_source = auth_source
        self.replica_set = replica_set
        self.min_pool_size = min_pool_size
        self.max_idle_time_ms = max_idle_time_ms
        self.wait_queue_timeout_ms = wait_queue_timeout_ms
        self.server_selection_timeout_ms = server_selection_timeout_ms
        self.extra_options = kwargs
    
    def get_connection_string(self) -> str:
        """获取连接字符串"""
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"
        else:
            auth_part = ""
        
        base = f"mongodb://{auth_part}{self.host}:{self.port}/{self.database}"
        
        params = []
        if self.auth_source:
            params.append(f"authSource={self.auth_source}")
        if self.replica_set:
            params.append(f"replicaSet={self.replica_set}")
        if self.ssl:
            params.append("ssl=true")
        
        if params:
            base += "?" + "&".join(params)
        
        return base


class MongoStorage(BaseStorage):
    """
    MongoDB存储实现
    
    特性：
    - 文档存储，适合非结构化数据
    - 支持嵌套文档和数组
    - 强大的查询和聚合能力
    - 水平扩展 (sharding)
    """
    
    def __init__(self, config: Optional[MongoConfig] = None):
        if config is None:
            config = MongoConfig()
        super().__init__(config)
        self.config: MongoConfig = config
        self._client = None
        self._db = None
    
    def connect(self) -> bool:
        """连接到MongoDB"""
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure
            
            connection_string = self.config.get_connection_string()
            
            self._client = MongoClient(
                connection_string,
                maxPoolSize=self.config.max_pool_size,
                minPoolSize=self.config.min_pool_size,
                maxIdleTimeMS=self.config.max_idle_time_ms,
                waitQueueTimeoutMS=self.config.wait_queue_timeout_ms,
                serverSelectionTimeoutMS=self.config.server_selection_timeout_ms
            )
            
            # 验证连接
            self._client.admin.command('ping')
            self._db = self._client[self.config.database]
            self._connected = True
            
            logger.info(f"Connected to MongoDB at {self.config.host}:{self.config.port}")
            return True
            
        except ImportError:
            logger.error("pymongo not installed. Run: pip install pymongo")
            return False
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            return False
    
    def disconnect(self) -> bool:
        """断开MongoDB连接"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._connected = False
            logger.info("Disconnected from MongoDB")
        return True
    
    def create_collection(self, name: str, schema: Optional[Dict] = None) -> bool:
        """创建集合"""
        try:
            if name not in self._db.list_collection_names():
                self._db.create_collection(name)
                logger.info(f"Created collection: {name}")
            
            # 如果提供了schema，创建验证规则
            if schema:
                self._db.command({
                    'collMod': name,
                    'validator': {'$jsonSchema': schema},
                    'validationLevel': 'moderate'
                })
            
            return True
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            return False
    
    def drop_collection(self, name: str) -> bool:
        """删除集合"""
        try:
            self._db.drop_collection(name)
            logger.info(f"Dropped collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop collection {name}: {e}")
            return False
    
    def insert(self, collection: str, record: DataRecord) -> str:
        """插入单条记录"""
        try:
            doc = record.to_dict()
            
            # 移除None的id，让MongoDB自动生成
            if doc.get("id") is None:
                del doc["id"]
            
            result = self._db[collection].insert_one(doc)
            record_id = str(result.inserted_id)
            record.id = record_id
            
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to insert record: {e}")
            raise
    
    def insert_many(self, collection: str, records: List[DataRecord]) -> List[str]:
        """批量插入记录"""
        try:
            from pymongo import InsertOne
            
            docs = []
            for record in records:
                doc = record.to_dict()
                if doc.get("id") is None:
                    del doc["id"]
                docs.append(doc)
            
            if not docs:
                return []
            
            result = self._db[collection].insert_many(docs)
            
            # 更新记录的id
            for i, record in enumerate(records):
                record.id = str(result.inserted_ids[i])
            
            return [str(oid) for oid in result.inserted_ids]
            
        except Exception as e:
            logger.error(f"Failed to insert records: {e}")
            raise
    
    def find_by_id(self, collection: str, record_id: str) -> Optional[DataRecord]:
        """根据ID查找记录"""
        try:
            from bson.objectid import ObjectId
            
            try:
                obj_id = ObjectId(record_id)
            except:
                # 如果不是有效的ObjectId，作为字符串查找
                doc = self._db[collection].find_one({"_id": record_id})
            else:
                doc = self._db[collection].find_one({"_id": obj_id})
            
            if doc:
                # 转换_id到id
                doc["id"] = str(doc.pop("_id"))
                return DataRecord.from_dict(doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to find record by id: {e}")
            return None
    
    def find(
        self,
        collection: str,
        filters: Optional[List[QueryFilter]] = None,
        sort: Optional[Dict[str, str]] = None,
        page: int = 1,
        page_size: int = 100
    ) -> QueryResult:
        """查询记录"""
        import time
        
        start_time = time.time()
        
        try:
            # 构建查询条件
            query = self._build_query(filters)
            
            # 获取总数
            total_count = self._db[collection].count_documents(query)
            
            # 构建游标
            cursor = self._db[collection].find(query)
            
            # 排序
            if sort:
                sort_list = []
                for field, direction in sort.items():
                    sort_list.append((field, 1 if direction == "asc" else -1))
                cursor = cursor.sort(sort_list)
            
            # 分页
            skip = (page - 1) * page_size
            cursor = cursor.skip(skip).limit(page_size)
            
            # 获取结果
            records = []
            for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                records.append(DataRecord.from_dict(doc))
            
            execution_time = (time.time() - start_time) * 1000
            
            return QueryResult(
                records=records,
                total_count=total_count,
                page=page,
                page_size=page_size,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Failed to find records: {e}")
            return QueryResult(records=[], total_count=0)
    
    def update(self, collection: str, record_id: str, data: Dict[str, Any]) -> bool:
        """更新记录"""
        try:
            from bson.objectid import ObjectId
            from pymongo import ReturnDocument
            
            try:
                obj_id = ObjectId(record_id)
            except:
                obj_id = record_id
            
            update_data = {
                "$set": {
                    "data": data,
                    "updated_at": datetime.now().isoformat()
                },
                "$inc": {"version": 1}
            }
            
            result = self._db[collection].update_one(
                {"_id": obj_id},
                update_data
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update record: {e}")
            return False
    
    def update_many(
        self,
        collection: str,
        filters: List[QueryFilter],
        data: Dict[str, Any]
    ) -> int:
        """批量更新记录"""
        try:
            query = self._build_query(filters)
            
            update_data = {
                "$set": {
                    "data": data,
                    "updated_at": datetime.now().isoformat()
                },
                "$inc": {"version": 1}
            }
            
            result = self._db[collection].update_many(query, update_data)
            return result.modified_count
            
        except Exception as e:
            logger.error(f"Failed to update records: {e}")
            return 0
    
    def delete(self, collection: str, record_id: str) -> bool:
        """删除记录"""
        try:
            from bson.objectid import ObjectId
            
            try:
                obj_id = ObjectId(record_id)
            except:
                obj_id = record_id
            
            result = self._db[collection].delete_one({"_id": obj_id})
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete record: {e}")
            return False
    
    def delete_many(self, collection: str, filters: List[QueryFilter]) -> int:
        """批量删除记录"""
        try:
            query = self._build_query(filters)
            result = self._db[collection].delete_many(query)
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete records: {e}")
            return 0
    
    def count(self, collection: str, filters: Optional[List[QueryFilter]] = None) -> int:
        """计数记录"""
        try:
            query = self._build_query(filters)
            return self._db[collection].count_documents(query)
        except Exception as e:
            logger.error(f"Failed to count records: {e}")
            return 0
    
    def aggregate(
        self,
        collection: str,
        pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """聚合查询"""
        try:
            return list(self._db[collection].aggregate(pipeline))
        except Exception as e:
            logger.error(f"Failed to aggregate: {e}")
            return []
    
    def create_index(
        self,
        collection: str,
        fields: List[str],
        index_type: str = "btree",
        unique: bool = False
    ) -> bool:
        """创建索引"""
        try:
            from pymongo import ASCENDING, DESCENDING, TEXT, HASHED
            
            if index_type == "text":
                # 文本索引
                self._db[collection].create_index(
                    [(f, TEXT) for f in fields],
                    unique=unique
                )
            elif index_type == "hash":
                # 哈希索引
                self._db[collection].create_index(
                    [(f, HASHED) for f in fields]
                )
            else:
                # B-tree索引 (默认升序)
                index_fields = [(f, ASCENDING) for f in fields]
                self._db[collection].create_index(
                    index_fields,
                    unique=unique
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def transaction(self) -> "MongoTransaction":
        """开始事务"""
        return MongoTransaction(self)
    
    def _build_query(self, filters: Optional[List[QueryFilter]]) -> Dict[str, Any]:
        """构建MongoDB查询条件"""
        if not filters:
            return {}
        
        query = {}
        for f in filters:
            mongo_op = self._map_operator(f.operator)
            if mongo_op:
                query[f.field] = {mongo_op: f.value}
            else:
                query[f.field] = f.value
        
        return query
    
    def _map_operator(self, op: str) -> Optional[str]:
        """映射操作符到MongoDB操作符"""
        mapping = {
            "eq": "$eq",
            "ne": "$ne",
            "gt": "$gt",
            "lt": "$lt",
            "gte": "$gte",
            "lte": "$lte",
            "in": "$in",
            "nin": "$nin",
            "regex": "$regex",
            "exists": "$exists"
        }
        return mapping.get(op)
    
    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = self._db.command("collStats", collection)
            return {
                "document_count": stats.get("count", 0),
                "size_bytes": stats.get("size", 0),
                "avg_obj_size": stats.get("avgObjSize", 0),
                "storage_size": stats.get("storageSize", 0),
                "index_count": len(stats.get("indexSizes", {})),
                "index_size": sum(stats.get("indexSizes", {}).values())
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def list_collections(self) -> List[str]:
        """列出所有集合"""
        try:
            return self._db.list_collection_names()
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def export_to_json(self, collection: str, output_path: str):
        """导出集合到JSON文件"""
        import json
        
        records = []
        for doc in self._db[collection].find():
            doc["id"] = str(doc.pop("_id"))
            records.append(doc)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, default=str)
        
        logger.info(f"Exported {len(records)} records to {output_path}")
    
    def import_from_json(self, collection: str, input_path: str):
        """从JSON文件导入数据"""
        import json
        
        with open(input_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        data_records = [DataRecord.from_dict(r) for r in records]
        self.insert_many(collection, data_records)
        
        logger.info(f"Imported {len(data_records)} records from {input_path}")


class MongoTransaction(StorageTransaction):
    """MongoDB事务"""
    
    def __init__(self, storage: MongoStorage):
        super().__init__(storage)
        self._session = None
    
    def begin(self):
        """开始事务"""
        if self.storage._client:
            self._session = self.storage._client.start_session()
            self._session.start_transaction()
            self._active = True
    
    def commit(self):
        """提交事务"""
        if self._session:
            self._session.commit_transaction()
            self._session.end_session()
            self._session = None
        self._active = False
    
    def rollback(self):
        """回滚事务"""
        if self._session:
            self._session.abort_transaction()
            self._session.end_session()
            self._session = None
        self._active = False
    
    @property
    def session(self):
        """获取会话"""
        return self._session
