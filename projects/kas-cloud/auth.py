"""
认证模块
"""
import hashlib
import os
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from database import get_db

# JWT 配置 - 从环境变量读取，避免重启后 token 失效
# 生产环境必须设置 JWT_SECRET 环境变量
JWT_SECRET = os.getenv("JWT_SECRET", "kas-cloud-dev-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """哈希密码"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """验证密码"""
    return hash_password(password) == hashed


def generate_api_key() -> str:
    """生成 API Key"""
    return f"kas_{secrets.token_urlsafe(32)}"


def create_access_token(user_id: int, username: str, role: str) -> str:
    """创建 JWT Token"""
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """解码 JWT Token"""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前用户"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    token_data = decode_token(credentials.credentials)
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, role, api_key, created_at FROM users WHERE id = ?",
            (token_data["sub"],)
        )
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=401, detail="User not found")
        
        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "role": row["role"],
            "api_key": row["api_key"],
            "created_at": row["created_at"]
        }


def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前用户（可选）"""
    if not credentials:
        return None
    
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None
