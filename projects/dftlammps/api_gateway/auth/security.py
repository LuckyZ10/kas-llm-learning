"""
认证授权模块
提供JWT令牌管理、OAuth2认证、密码哈希等功能
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, APIKeyHeader
from pydantic import BaseModel
import secrets
import hashlib

# JWT配置
SECRET_KEY = "your-secret-key-change-in-production"  # 生产环境应使用环境变量
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# 密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    scopes={
        "read": "Read access",
        "write": "Write access",
        "admin": "Admin access",
    }
)

# API Key scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class Token(BaseModel):
    """令牌模型"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """令牌数据"""
    username: Optional[str] = None
    scopes: List[str] = []
    user_id: Optional[int] = None


class User(BaseModel):
    """用户模型"""
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    role: str = "researcher"
    status: str = "active"
    scopes: List[str] = []
    quota_limit: int = 1000
    quota_used: int = 0
    
    class Config:
        from_attributes = True


class UserInDB(User):
    """数据库用户模型"""
    hashed_password: str
    api_key: Optional[str] = None


# 模拟用户数据库 - 生产环境应使用真实数据库
USERS_DB: Dict[str, UserInDB] = {}
API_KEYS_DB: Dict[str, int] = {}  # api_key -> user_id mapping


def init_default_users():
    """初始化默认用户"""
    global USERS_DB, API_KEYS_DB
    
    # 创建管理员用户
    admin_user = UserInDB(
        id=1,
        username="admin",
        email="admin@dftlammps.org",
        full_name="Administrator",
        role="admin",
        status="active",
        scopes=["read", "write", "admin"],
        hashed_password=get_password_hash("admin123"),  # 生产环境应修改
        quota_limit=10000,
        quota_used=0,
    )
    USERS_DB["admin"] = admin_user
    
    # 创建演示用户
    demo_user = UserInDB(
        id=2,
        username="demo",
        email="demo@dftlammps.org",
        full_name="Demo User",
        role="researcher",
        status="active",
        scopes=["read", "write"],
        hashed_password=get_password_hash("demo123"),  # 生产环境应修改
        quota_limit=1000,
        quota_used=0,
    )
    USERS_DB["demo"] = demo_user
    
    # 创建API客户端用户
    api_user = UserInDB(
        id=3,
        username="api_client",
        email="api@dftlammps.org",
        full_name="API Client",
        role="api_client",
        status="active",
        scopes=["read"],
        hashed_password=get_password_hash("api123"),
        api_key="dftlammps_demo_api_key_12345",
        quota_limit=5000,
        quota_used=0,
    )
    USERS_DB["api_client"] = api_user
    API_KEYS_DB["dftlammps_demo_api_key_12345"] = 3


# 初始化默认用户
init_default_users()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """获取密码哈希"""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """获取用户"""
    return USERS_DB.get(username)


def get_user_by_id(user_id: int) -> Optional[UserInDB]:
    """根据ID获取用户"""
    for user in USERS_DB.values():
        if user.id == user_id:
            return user
    return None


def get_user_by_api_key(api_key: str) -> Optional[UserInDB]:
    """根据API Key获取用户"""
    user_id = API_KEYS_DB.get(api_key)
    if user_id:
        return get_user_by_id(user_id)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """认证用户"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """创建刷新令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """解码令牌"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


async def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> User:
    """获取当前用户 (OAuth2)"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, scopes=payload.get("scopes", []))
    except JWTError:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    
    return User(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        status=user.status,
        scopes=token_data.scopes or user.scopes,
        quota_limit=user.quota_limit,
        quota_used=user.quota_used,
    )


async def get_current_user_api_key(
    api_key: Optional[str] = Depends(api_key_header)
) -> Optional[User]:
    """获取当前用户 (API Key)"""
    if not api_key:
        return None
    
    user = get_user_by_api_key(api_key)
    if not user:
        return None
    
    return User(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        status=user.status,
        scopes=user.scopes,
        quota_limit=user.quota_limit,
        quota_used=user.quota_used,
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """获取当前活跃用户"""
    if current_user.status != "active":
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_user_or_api_key(
    token_user: Optional[User] = Depends(get_current_user),
    api_key_user: Optional[User] = Depends(get_current_user_api_key),
) -> User:
    """获取当前用户 (支持OAuth2或API Key)"""
    if token_user:
        return token_user
    if api_key_user:
        return api_key_user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required (OAuth2 token or API Key)",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_scopes(required_scopes: List[str]):
    """装饰器：要求特定权限"""
    def decorator(func):
        async def wrapper(*args, current_user: User = Depends(get_current_user_or_api_key), **kwargs):
            user_scopes = set(current_user.scopes)
            required = set(required_scopes)
            
            if not required.issubset(user_scopes):
                missing = required - user_scopes
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required scopes: {missing}"
                )
            
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


def require_role(required_roles: List[str]):
    """装饰器：要求特定角色"""
    def decorator(func):
        async def wrapper(*args, current_user: User = Depends(get_current_user_or_api_key), **kwargs):
            if current_user.role not in required_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required role: one of {required_roles}"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


def generate_api_key() -> str:
    """生成新的API Key"""
    key = secrets.token_urlsafe(32)
    prefix = "dftlammps_"
    return prefix + key


def hash_api_key(api_key: str) -> str:
    """哈希API Key (用于存储)"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def refresh_access_token(refresh_token: str) -> Token:
    """使用刷新令牌获取新的访问令牌"""
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        token_type = payload.get("type")
        
        if token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        user = get_user(username)
        if not user or user.status != "active":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # 创建新的访问令牌
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": user.username,
                "scopes": user.scopes,
                "user_id": user.id
            },
            expires_delta=access_token_expires
        )
        
        # 创建新的刷新令牌
        new_refresh_token = create_refresh_token(
            data={"sub": user.username}
        )
        
        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
