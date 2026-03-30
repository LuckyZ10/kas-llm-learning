"""
KAS Cloud API Server
云端市场后端服务
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class UserRole(str, Enum):
    """用户角色"""
    USER = "user"
    ADMIN = "admin"


class User(BaseModel):
    """用户模型"""
    id: int
    username: str
    email: str
    role: UserRole = UserRole.USER
    created_at: datetime
    api_key: Optional[str] = None
    
    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """创建用户请求"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    """登录请求"""
    username: str
    password: str


class AgentPackage(BaseModel):
    """Agent 包模型"""
    id: int
    name: str
    version: str
    description: Optional[str] = None
    author_id: int
    author_name: str
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    tags: List[str] = []
    capabilities: List[str] = []
    file_size: int = 0
    file_hash: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class AgentPackageCreate(BaseModel):
    """发布 Agent 请求"""
    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    description: Optional[str] = None
    tags: List[str] = []
    capabilities: List[str] = []


class AgentPackageSearch(BaseModel):
    """搜索结果"""
    id: int
    name: str
    version: str
    description: Optional[str] = None
    author_name: str
    downloads: int
    rating: float
    tags: List[str] = []
    created_at: datetime


class Token(BaseModel):
    """认证令牌"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class SearchQuery(BaseModel):
    """搜索查询"""
    q: Optional[str] = None
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    sort_by: str = "downloads"  # downloads, rating, created_at
    order: str = "desc"
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class RatingCreate(BaseModel):
    """评分请求"""
    score: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class DownloadResponse(BaseModel):
    """下载响应"""
    package_id: int
    package_name: str
    version: str
    download_url: str
    expires_at: datetime
