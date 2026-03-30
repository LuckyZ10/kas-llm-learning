"""
PostgreSQL Storage - PostgreSQL存储实现
======================================
基于psycopg2的关系型存储实现，适合结构化查询和复杂关联。
"""

from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
import json
import logging

from .base_storage import (
    BaseStorage, StorageConfig, StorageType,
    QueryFilter, DataRecord, QueryResult, StorageTransaction
)

logger = logging.getLogger(__name__)


@dataclass
class PostgresConfig(StorageConfig):
    """PostgreSQL配置"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "dftlammps",
        username: str = "postgres",
        password: Optional[str] = None,
        ssl_mode: str = "prefer",
        connect_timeout: int = 30,
        application_name: str = "dftlammps",
        **kwargs
    ):
        super().__init__(
            storage_type=StorageType.RELATIONAL,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password
        )
        self.ssl_mode = ssl_mode
        self.connect_timeout = connect_timeout
        self.application_name = application_name
        self.extra_options = kwargs


class PostgresStorage(BaseStorage):
    """
    PostgreSQL存储实现
    
    特性：
    - ACID事务支持
    - 复杂查询和JOIN操作
    - JSON/JSONB字段支持
    - 全文搜索
    - 时序数据扩展 (TimescaleDB)
    - 向量扩展 (pgvector)
    """
    
    def __init__(self, config: Optional[PostgresConfig] = None):
        if config is None:
            config = PostgresConfig()
        super().__init__(config)
        self.config: PostgresConfig = config
        self._pool = None
        self._connection = None
    
    def connect(self) -> bool:
        """连接到PostgreSQL"""
        try:
            import psycopg2
            from psycopg2 import pool
            
            conn_params = {
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "user": self.config.username,
                "password": self.config.password,
                "sslmode": self.config.ssl_mode,
                "connect_timeout": self.config.connect_timeout,
                "application_name": self.config.application_name
            }
            
            # 移除None值
            conn_params = {k: v for k, v in conn_params.items() if v is not None}
            
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=self.config.max_pool_size,
                **conn_params
            )
            
            # 测试连接
            conn = self._pool.getconn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            self._pool.putconn(conn)
            
            self._connected = True
            logger.info(f"Connected to PostgreSQL at {self.config.host}:{self.config.port}")
            return True
            
        except ImportError:
            logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def disconnect(self) -> bool:
        """断开PostgreSQL连接"""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            self._connected = False
            logger.info("Disconnected from PostgreSQL")
        return True
    
    @contextmanager
    def _get_connection(self):
        """获取连接的上下文管理器"""
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)
    
    def create_collection(self, name: str, schema: Optional[Dict] = None) -> bool:
        """创建表"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # 检查表是否存在
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                    """, (name,))
                    exists = cur.fetchone()[0]
                    
                    if not exists:
                        # 创建表结构
                        create_sql = f"""
                            CREATE TABLE {name} (
                                id SERIAL PRIMARY KEY,
                                data JSONB NOT NULL DEFAULT '{{}}',
                                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                                version INTEGER DEFAULT 1,
                                checksum VARCHAR(32),
                                tags TEXT[] DEFAULT ARRAY[]::TEXT[],
                                source VARCHAR(255),
                                search_vector tsvector
                            )
                        """
                        cur.execute(create_sql)
                        
                        # 创建GIN索引用于JSONB查询
                        cur.execute(f"""
                            CREATE INDEX idx_{name}_data_gin ON {name} USING GIN (data)
                        """)
                        
                        # 创建全文搜索索引
                        cur.execute(f"""
                            CREATE INDEX idx_{name}_search ON {name} USING GIN (search_vector)
                        """)
                        
                        # 创建更新时间触发器
                        cur.execute(f"""
                            CREATE OR REPLACE FUNCTION update_{name}_updated_at()
                            RETURNS TRIGGER AS $$
                            BEGIN
                                NEW.updated_at = NOW();
                                RETURN NEW;
                            END;
                            $$ LANGUAGE plpgsql;
                            
                            DROP TRIGGER IF EXISTS trigger_{name}_updated_at ON {name};
                            CREATE TRIGGER trigger_{name}_updated_at
                                BEFORE UPDATE ON {name}
                                FOR EACH ROW
                                EXECUTE FUNCTION update_{name}_updated_at();
                        """)
                        
                        conn.commit()
                        logger.info(f"Created table: {name}")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to create table {name}: {e}")
            return False
    
    def drop_collection(self, name: str) -> bool:
        """删除表"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
                    conn.commit()
                    logger.info(f"Dropped table: {name}")
                    return True
        except Exception as e:
            logger.error(f"Failed to drop table {name}: {e}")
            return False
    
    def insert(self, collection: str, record: DataRecord) -> str:
        """插入记录"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        INSERT INTO {collection} 
                        (data, created_at, updated_at, version, checksum, tags, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        json.dumps(record.data),
                        record.created_at or datetime.now(),
                        record.updated_at or datetime.now(),
                        record.version,
                        record.checksum,
                        record.tags,
                        record.source
                    ))
                    
                    record_id = str(cur.fetchone()[0])
                    record.id = record_id
                    
                    conn.commit()
                    return record_id
                    
        except Exception as e:
            logger.error(f"Failed to insert record: {e}")
            raise
    
    def insert_many(self, collection: str, records: List[DataRecord]) -> List[str]:
        """批量插入记录"""
        try:
            import psycopg2.extras
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # 使用execute_values进行批量插入
                    values = [
                        (
                            json.dumps(r.data),
                            r.created_at or datetime.now(),
                            r.updated_at or datetime.now(),
                            r.version,
                            r.checksum,
                            r.tags,
                            r.source
                        )
                        for r in records
                    ]
                    
                    psycopg2.extras.execute_values(
                        cur,
                        f"""
                        INSERT INTO {collection} 
                        (data, created_at, updated_at, version, checksum, tags, source)
                        VALUES %s
                        RETURNING id
                        """,
                        values
                    )
                    
                    ids = [str(row[0]) for row in cur.fetchall()]
                    
                    # 更新记录ID
                    for i, record in enumerate(records):
                        record.id = ids[i]
                    
                    conn.commit()
                    return ids
                    
        except Exception as e:
            logger.error(f"Failed to insert records: {e}")
            raise
    
    def find_by_id(self, collection: str, record_id: str) -> Optional[DataRecord]:
        """根据ID查找记录"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT id, data, created_at, updated_at, version, checksum, tags, source
                        FROM {collection}
                        WHERE id = %s
                    """, (record_id,))
                    
                    row = cur.fetchone()
                    if row:
                        return self._row_to_record(row)
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to find record: {e}")
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
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # 构建WHERE条件
                    where_clause, params = self._build_where_clause(filters)
                    
                    # 计数
                    count_sql = f"SELECT COUNT(*) FROM {collection}"
                    if where_clause:
                        count_sql += f" WHERE {where_clause}"
                    cur.execute(count_sql, params)
                    total_count = cur.fetchone()[0]
                    
                    # 查询
                    query_sql = f"""
                        SELECT id, data, created_at, updated_at, version, checksum, tags, source
                        FROM {collection}
                    """
                    if where_clause:
                        query_sql += f" WHERE {where_clause}"
                    
                    # 排序
                    if sort:
                        order_clauses = []
                        for field, direction in sort.items():
                            # 处理JSONB字段排序
                            if field.startswith("data."):
                                json_path = field.replace("data.", "")
                                order_clauses.append(
                                    f"data->>'{json_path}' {'ASC' if direction == 'asc' else 'DESC'}"
                                )
                            else:
                                order_clauses.append(
                                    f"{field} {'ASC' if direction == 'asc' else 'DESC'}"
                                )
                        query_sql += " ORDER BY " + ", ".join(order_clauses)
                    else:
                        query_sql += " ORDER BY created_at DESC"
                    
                    # 分页
                    offset = (page - 1) * page_size
                    query_sql += f" LIMIT {page_size} OFFSET {offset}"
                    
                    cur.execute(query_sql, params)
                    rows = cur.fetchall()
                    
                    records = [self._row_to_record(row) for row in rows]
                    
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
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        UPDATE {collection}
                        SET data = %s,
                            updated_at = NOW(),
                            version = version + 1
                        WHERE id = %s
                    """, (json.dumps(data), record_id))
                    
                    conn.commit()
                    return cur.rowcount > 0
                    
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
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    where_clause, params = self._build_where_clause(filters)
                    
                    sql = f"""
                        UPDATE {collection}
                        SET data = %s,
                            updated_at = NOW(),
                            version = version + 1
                    """
                    if where_clause:
                        sql += f" WHERE {where_clause}"
                    
                    cur.execute(sql, (json.dumps(data),) + tuple(params))
                    conn.commit()
                    return cur.rowcount
                    
        except Exception as e:
            logger.error(f"Failed to update records: {e}")
            return 0
    
    def delete(self, collection: str, record_id: str) -> bool:
        """删除记录"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        DELETE FROM {collection}
                        WHERE id = %s
                    """, (record_id,))
                    
                    conn.commit()
                    return cur.rowcount > 0
                    
        except Exception as e:
            logger.error(f"Failed to delete record: {e}")
            return False
    
    def delete_many(self, collection: str, filters: List[QueryFilter]) -> int:
        """批量删除记录"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    where_clause, params = self._build_where_clause(filters)
                    
                    sql = f"DELETE FROM {collection}"
                    if where_clause:
                        sql += f" WHERE {where_clause}"
                    
                    cur.execute(sql, params)
                    conn.commit()
                    return cur.rowcount
                    
        except Exception as e:
            logger.error(f"Failed to delete records: {e}")
            return 0
    
    def count(self, collection: str, filters: Optional[List[QueryFilter]] = None) -> int:
        """计数记录"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    where_clause, params = self._build_where_clause(filters)
                    
                    sql = f"SELECT COUNT(*) FROM {collection}"
                    if where_clause:
                        sql += f" WHERE {where_clause}"
                    
                    cur.execute(sql, params)
                    return cur.fetchone()[0]
                    
        except Exception as e:
            logger.error(f"Failed to count records: {e}")
            return 0
    
    def aggregate(
        self,
        collection: str,
        pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """聚合查询 (使用PostgreSQL的聚合函数)"""
        try:
            # 简化实现，实际应解析pipeline
            results = []
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # 示例：简单的分组统计
                    cur.execute(f"""
                        SELECT 
                            jsonb_object_keys(data) as key,
                            COUNT(*) as count
                        FROM {collection}
                        GROUP BY jsonb_object_keys(data)
                    """)
                    
                    for row in cur.fetchall():
                        results.append({"key": row[0], "count": row[1]})
            
            return results
            
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
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    index_name = f"idx_{collection}_{'_'.join(fields)}"
                    
                    if index_type == "gin":
                        # GIN索引用于JSONB
                        cur.execute(f"""
                            CREATE INDEX {index_name} 
                            ON {collection} USING GIN (data)
                        """)
                    elif index_type == "text":
                        # 全文搜索索引
                        cur.execute(f"""
                            CREATE INDEX {index_name}
                            ON {collection} USING GIN (to_tsvector('english', data::text))
                        """)
                    else:
                        # B-tree索引
                        index_fields = []
                        for field in fields:
                            if field.startswith("data."):
                                json_path = field.replace("data.", "")
                                index_fields.append(f"(data->>'{json_path}')")
                            else:
                                index_fields.append(field)
                        
                        unique_str = "UNIQUE" if unique else ""
                        cur.execute(f"""
                            CREATE {unique_str} INDEX {index_name}
                            ON {collection} ({', '.join(index_fields)})
                        """)
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def transaction(self) -> "PostgresTransaction":
        """开始事务"""
        return PostgresTransaction(self)
    
    def _build_where_clause(self, filters: Optional[List[QueryFilter]]) -> Tuple[str, tuple]:
        """构建WHERE子句"""
        if not filters:
            return "", ()
        
        conditions = []
        params = []
        
        for f in filters:
            if f.field.startswith("data."):
                # JSONB字段查询
                json_path = f.field.replace("data.", "")
                op = self._map_operator(f.operator)
                conditions.append(f"data->>'{json_path}' {op} %s")
                params.append(f.value)
            else:
                # 普通字段查询
                op = self._map_operator(f.operator)
                conditions.append(f"{f.field} {op} %s")
                params.append(f.value)
        
        return " AND ".join(conditions), tuple(params)
    
    def _map_operator(self, op: str) -> str:
        """映射操作符到SQL操作符"""
        mapping = {
            "eq": "=",
            "ne": "!=",
            "gt": ">",
            "lt": "<",
            "gte": ">=",
            "lte": "<=",
            "in": "IN",
            "nin": "NOT IN",
            "regex": "~",
            "exists": "IS NOT NULL"
        }
        return mapping.get(op, "=")
    
    def _row_to_record(self, row: Tuple) -> DataRecord:
        """将数据库行转换为DataRecord"""
        return DataRecord(
            id=str(row[0]),
            data=row[1] if isinstance(row[1], dict) else json.loads(row[1]),
            created_at=row[2],
            updated_at=row[3],
            version=row[4],
            checksum=row[5],
            tags=list(row[6]) if row[6] else [],
            source=row[7]
        )
    
    def execute_sql(self, sql: str, params: Optional[tuple] = None) -> List[Tuple]:
        """执行原始SQL"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params or ())
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to execute SQL: {e}")
            return []
    
    def enable_pgvector(self) -> bool:
        """启用pgvector扩展"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    conn.commit()
                    logger.info("Enabled pgvector extension")
                    return True
        except Exception as e:
            logger.error(f"Failed to enable pgvector: {e}")
            return False
    
    def add_vector_column(self, collection: str, column: str, dimensions: int) -> bool:
        """添加向量列"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        ALTER TABLE {collection}
                        ADD COLUMN IF NOT EXISTS {column} vector({dimensions})
                    """)
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Failed to add vector column: {e}")
            return False


class PostgresTransaction(StorageTransaction):
    """PostgreSQL事务"""
    
    def __init__(self, storage: PostgresStorage):
        super().__init__(storage)
        self._conn = None
    
    def begin(self):
        """开始事务"""
        self._conn = self.storage._pool.getconn()
        self._active = True
    
    def commit(self):
        """提交事务"""
        if self._conn:
            self._conn.commit()
            self.storage._pool.putconn(self._conn)
            self._conn = None
        self._active = False
    
    def rollback(self):
        """回滚事务"""
        if self._conn:
            self._conn.rollback()
            self.storage._pool.putconn(self._conn)
            self._conn = None
        self._active = False
    
    @property
    def connection(self):
        """获取连接"""
        return self._conn
