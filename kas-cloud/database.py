"""
数据库模块
"""
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

# 数据库路径 - 使用用户主目录，避免 /tmp 重启丢失和 Windows 不兼容问题
DB_DIR = Path(os.getenv("KAS_CLOUD_DB", Path.home() / ".kas" / "cloud"))
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "kas_cloud.db"


def init_db():
    """初始化数据库"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                api_key TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Agent 包表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_packages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT,
                author_id INTEGER NOT NULL,
                downloads INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0,
                rating_count INTEGER DEFAULT 0,
                tags TEXT,  -- JSON 数组
                capabilities TEXT,  -- JSON 数组
                file_path TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                file_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, version),
                FOREIGN KEY (author_id) REFERENCES users(id)
            )
        ''')
        
        # 评分表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                package_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                score INTEGER NOT NULL CHECK(score >= 1 AND score <= 5),
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(package_id, user_id),
                FOREIGN KEY (package_id) REFERENCES agent_packages(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # 下载记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS downloads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                package_id INTEGER NOT NULL,
                user_id INTEGER,
                ip_address TEXT,
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (package_id) REFERENCES agent_packages(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_packages_name ON agent_packages(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_packages_author ON agent_packages(author_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_packages_downloads ON agent_packages(downloads DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_packages_rating ON agent_packages(rating DESC)')
        
        conn.commit()
        print(f"✅ Database initialized at {DB_PATH}")


@contextmanager
def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_db_connection():
    """获取数据库连接（用于 FastAPI Depends）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
