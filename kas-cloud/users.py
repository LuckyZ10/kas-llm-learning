"""
用户 API 模块
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from database import get_db
from auth import (
    hash_password, verify_password, generate_api_key,
    create_access_token, get_current_user
)
from models import UserCreate, UserLogin, User, Token

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.post("/register", response_model=User)
def register(user: UserCreate):
    """用户注册"""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 检查用户名是否已存在
        cursor.execute("SELECT id FROM users WHERE username = ?", (user.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="Username already exists")
        
        # 检查邮箱是否已存在
        cursor.execute("SELECT id FROM users WHERE email = ?", (user.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="Email already exists")
        
        # 创建用户
        password_hash = hash_password(user.password)
        api_key = generate_api_key()
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, api_key)
            VALUES (?, ?, ?, ?)
        ''', (user.username, user.email, password_hash, api_key))
        
        conn.commit()
        user_id = cursor.lastrowid
        
        # 获取创建的用户信息
        cursor.execute(
            "SELECT id, username, email, role, api_key, created_at FROM users WHERE id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "role": row["role"],
            "api_key": row["api_key"],
            "created_at": row["created_at"]
        }


@router.post("/login", response_model=Token)
def login(user: UserLogin):
    """用户登录"""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 查找用户
        cursor.execute(
            "SELECT id, username, email, password_hash, role FROM users WHERE username = ?",
            (user.username,)
        )
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # 验证密码
        if not verify_password(user.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # 创建 Token
        token = create_access_token(
            user_id=row["id"],
            username=row["username"],
            role=row["role"]
        )
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": 3600
        }


@router.get("/me", response_model=User)
def get_me(current_user: dict = Depends(get_current_user)):
    """获取当前用户信息"""
    return current_user


@router.post("/api-key/refresh")
def refresh_api_key(current_user: dict = Depends(get_current_user)):
    """刷新 API Key"""
    
    new_api_key = generate_api_key()
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET api_key = ? WHERE id = ?",
            (new_api_key, current_user["id"])
        )
        conn.commit()
    
    return {"api_key": new_api_key}
