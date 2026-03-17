"""
市场 API 模块
"""
import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse

from database import get_db, get_db_connection
from auth import get_current_user, get_current_user_optional
from models import (
    AgentPackage, AgentPackageCreate, AgentPackageSearch,
    SearchQuery, RatingCreate, User
)

router = APIRouter(prefix="/api/v1/market", tags=["market"])

# 文件存储路径 - 使用用户主目录，避免 /tmp 重启丢失
STORAGE_PATH = Path(os.getenv("KAS_STORAGE_PATH", Path.home() / ".kas" / "packages"))
STORAGE_PATH.mkdir(parents=True, exist_ok=True)


@router.post("/publish", response_model=AgentPackage)
def publish_package(
    package: AgentPackageCreate,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """发布 Agent 包"""
    
    # 验证文件类型
    if not file.filename.endswith('.kas-agent'):
        raise HTTPException(status_code=400, detail="File must be .kas-agent format")
    
    # 读取文件内容
    content = file.file.read()
    file_size = len(content)
    file_hash = hashlib.sha256(content).hexdigest()
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 检查是否已存在同名同版本
        cursor.execute(
            "SELECT id FROM agent_packages WHERE name = ? AND version = ?",
            (package.name, package.version)
        )
        if cursor.fetchone():
            raise HTTPException(
                status_code=409,
                detail=f"Package {package.name} v{package.version} already exists"
            )
        
        # 保存文件
        filename = f"{package.name}-{package.version}.kas-agent"
        file_path = STORAGE_PATH / filename
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # 插入数据库
        cursor.execute('''
            INSERT INTO agent_packages 
            (name, version, description, author_id, tags, capabilities, 
             file_path, file_size, file_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            package.name,
            package.version,
            package.description,
            current_user["id"],
            json.dumps(package.tags),
            json.dumps(package.capabilities),
            str(file_path),
            file_size,
            file_hash
        ))
        
        conn.commit()
        package_id = cursor.lastrowid
        
        # 获取完整数据返回
        cursor.execute('''
            SELECT p.*, u.username as author_name 
            FROM agent_packages p
            JOIN users u ON p.author_id = u.id
            WHERE p.id = ?
        ''', (package_id,))
        row = cursor.fetchone()
        
        return _row_to_package(row)


@router.get("/search", response_model=List[AgentPackageSearch])
def search_packages(
    q: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    author: Optional[str] = None,
    sort_by: str = "downloads",
    order: str = "desc",
    limit: int = 20,
    offset: int = 0
):
    """搜索 Agent 包"""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 构建查询
        query = '''
            SELECT p.id, p.name, p.version, p.description, 
                   u.username as author_name, p.downloads, p.rating, 
                   p.tags, p.created_at
            FROM agent_packages p
            JOIN users u ON p.author_id = u.id
            WHERE 1=1
        '''
        params = []
        
        if q:
            query += " AND (p.name LIKE ? OR p.description LIKE ?)"
            params.extend([f"%{q}%", f"%{q}%"])
        
        if tags:
            for tag in tags:
                query += " AND p.tags LIKE ?"
                params.append(f'%"{tag}"%')
        
        if author:
            query += " AND u.username = ?"
            params.append(author)
        
        # 排序
        order_sql = "DESC" if order == "desc" else "ASC"
        if sort_by in ["downloads", "rating", "created_at"]:
            query += f" ORDER BY p.{sort_by} {order_sql}"
        else:
            query += " ORDER BY p.downloads DESC"
        
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "name": row["name"],
                "version": row["version"],
                "description": row["description"],
                "author_name": row["author_name"],
                "downloads": row["downloads"],
                "rating": row["rating"],
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "created_at": row["created_at"]
            })
        
        return results


@router.get("/packages/{package_id}", response_model=AgentPackage)
def get_package(package_id: int):
    """获取 Agent 包详情"""
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.*, u.username as author_name 
            FROM agent_packages p
            JOIN users u ON p.author_id = u.id
            WHERE p.id = ?
        ''', (package_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Package not found")
        
        return _row_to_package(row)


@router.post("/packages/{package_id}/download")
def download_package(
    package_id: int,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """下载 Agent 包"""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 获取包信息
        cursor.execute(
            "SELECT file_path, name, version FROM agent_packages WHERE id = ?",
            (package_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Package not found")
        
        # 更新下载计数
        cursor.execute(
            "UPDATE agent_packages SET downloads = downloads + 1 WHERE id = ?",
            (package_id,)
        )
        
        # 记录下载
        user_id = current_user["id"] if current_user else None
        cursor.execute('''
            INSERT INTO downloads (package_id, user_id)
            VALUES (?, ?)
        ''', (package_id, user_id))
        
        conn.commit()
        
        file_path = row["file_path"]
        filename = f"{row['name']}-{row['version']}.kas-agent"
        
        return FileResponse(
            file_path,
            filename=filename,
            media_type="application/octet-stream"
        )


@router.post("/packages/{package_id}/rate")
def rate_package(
    package_id: int,
    rating: RatingCreate,
    current_user: dict = Depends(get_current_user)
):
    """给 Agent 包评分"""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 检查包是否存在
        cursor.execute("SELECT id FROM agent_packages WHERE id = ?", (package_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Package not found")
        
        # 插入或更新评分
        cursor.execute('''
            INSERT INTO ratings (package_id, user_id, score, comment)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(package_id, user_id) 
            DO UPDATE SET score = excluded.score, comment = excluded.comment
        ''', (package_id, current_user["id"], rating.score, rating.comment))
        
        # 重新计算平均评分
        cursor.execute('''
            UPDATE agent_packages 
            SET rating = (
                SELECT AVG(score) FROM ratings WHERE package_id = ?
            ),
            rating_count = (
                SELECT COUNT(*) FROM ratings WHERE package_id = ?
            )
            WHERE id = ?
        ''', (package_id, package_id, package_id))
        
        conn.commit()
        
        return {"message": "Rating submitted successfully"}


@router.delete("/packages/{package_id}")
def delete_package(
    package_id: int,
    current_user: dict = Depends(get_current_user)
):
    """删除 Agent 包（仅作者或管理员）"""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 获取包信息
        cursor.execute(
            "SELECT author_id, file_path FROM agent_packages WHERE id = ?",
            (package_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Package not found")
        
        # 权限检查
        if row["author_id"] != current_user["id"] and current_user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Permission denied")
        
        # 删除文件
        file_path = Path(row["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # 删除数据库记录
        cursor.execute("DELETE FROM agent_packages WHERE id = ?", (package_id,))
        conn.commit()
        
        return {"message": "Package deleted successfully"}


def _row_to_package(row) -> AgentPackage:
    """数据库行转换为模型"""
    return AgentPackage(
        id=row["id"],
        name=row["name"],
        version=row["version"],
        description=row["description"],
        author_id=row["author_id"],
        author_name=row["author_name"],
        downloads=row["downloads"],
        rating=row["rating"],
        rating_count=row["rating_count"],
        tags=json.loads(row["tags"]) if row["tags"] else [],
        capabilities=json.loads(row["capabilities"]) if row["capabilities"] else [],
        file_size=row["file_size"],
        file_hash=row["file_hash"],
        created_at=row["created_at"],
        updated_at=row["updated_at"]
    )
