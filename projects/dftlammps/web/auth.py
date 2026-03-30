#!/usr/bin/env python3
"""
Authentication & Authorization Module
=====================================
User authentication and permission management for DFT-LAMMPS Web API.

Features:
- JWT-based authentication
- Role-based access control (RBAC)
- API key management
- User management

Author: DFT-LAMMPS Web Team
Version: 1.0.0
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
import json
import logging

# JWT
from jose import JWTError, jwt
from passlib.context import CryptContext

# Pydantic
from pydantic import BaseModel, Field, validator, EmailStr

# Redis for session storage
import redis

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
API_KEY_PREFIX = "dft_"

# Redis for session storage
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# =============================================================================
# Enums
# =============================================================================

class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"              # Full system access
    RESEARCHER = "researcher"    # Can submit jobs, view own results
    GUEST = "guest"              # Limited access, view only
    API = "api"                  # Service account for API access

class Permission(str, Enum):
    """Permissions for fine-grained access control."""
    # Task permissions
    TASK_CREATE = "task:create"
    TASK_READ = "task:read"
    TASK_READ_ALL = "task:read:all"
    TASK_UPDATE = "task:update"
    TASK_DELETE = "task:delete"
    TASK_CANCEL = "task:cancel"
    
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Admin
    ADMIN_ACCESS = "admin:access"
    SYSTEM_CONFIG = "system:config"
    QUEUE_MANAGE = "queue:manage"

# =============================================================================
# Pydantic Models
# =============================================================================

class UserBase(BaseModel):
    """Base user model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = Field(default=None, max_length=100)
    institution: Optional[str] = Field(default=None, max_length=100)
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum() and '_' not in v and '-' not in v:
            raise ValueError('Username must be alphanumeric with underscores or hyphens')
        return v

class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8, max_length=100)
    role: UserRole = UserRole.RESEARCHER

class UserUpdate(BaseModel):
    """User update model."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(default=None, max_length=100)
    institution: Optional[str] = Field(default=None, max_length=100)
    password: Optional[str] = Field(default=None, min_length=8)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

class User(UserBase):
    """User model with metadata."""
    id: str
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    api_key: Optional[str] = None
    
    class Config:
        from_attributes = True

class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str

class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds

class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    role: Optional[UserRole] = None
    permissions: List[str] = []

class APIKey(BaseModel):
    """API key model."""
    key_id: str
    name: str
    key: str  # Only shown once on creation
    user_id: str
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True

class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str

class PasswordResetRequest(BaseModel):
    """Password reset request model."""
    email: EmailStr

class PasswordReset(BaseModel):
    """Password reset model."""
    token: str
    new_password: str = Field(..., min_length=8)

# =============================================================================
# Role-Permission Mapping
# =============================================================================

ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.TASK_CREATE, Permission.TASK_READ, Permission.TASK_READ_ALL,
        Permission.TASK_UPDATE, Permission.TASK_DELETE, Permission.TASK_CANCEL,
        Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE, Permission.USER_DELETE,
        Permission.ADMIN_ACCESS, Permission.SYSTEM_CONFIG, Permission.QUEUE_MANAGE
    ],
    UserRole.RESEARCHER: [
        Permission.TASK_CREATE, Permission.TASK_READ, Permission.TASK_UPDATE, Permission.TASK_CANCEL
    ],
    UserRole.GUEST: [
        Permission.TASK_READ
    ],
    UserRole.API: [
        Permission.TASK_CREATE, Permission.TASK_READ, Permission.TASK_CANCEL
    ]
}

# =============================================================================
# User Database (Replace with proper database in production)
# =============================================================================

class UserDatabase:
    """User database interface using Redis."""
    
    USER_PREFIX = "dftlammps:user:"
    USERNAME_INDEX = "dftlammps:users:by_username"
    API_KEY_PREFIX = "dftlammps:apikey:"
    SESSION_PREFIX = "dftlammps:session:"
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def _user_key(self, user_id: str) -> str:
        return f"{self.USER_PREFIX}{user_id}"
    
    def _api_key(self, key_id: str) -> str:
        return f"{self.API_KEY_PREFIX}{key_id}"
    
    def _session_key(self, session_id: str) -> str:
        return f"{self.SESSION_PREFIX}{session_id}"
    
    def create_user(self, user_data: Dict) -> str:
        """Create a new user."""
        user_id = secrets.token_hex(16)
        user_data['id'] = user_id
        user_data['created_at'] = datetime.utcnow().isoformat()
        
        # Store user data
        self.redis.hset(self._user_key(user_id), mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in user_data.items()
        })
        
        # Index by username
        self.redis.hset(self.USERNAME_INDEX, user_data['username'], user_id)
        
        logger.info(f"User created: {user_data['username']} ({user_id})")
        return user_id
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by ID."""
        data = self.redis.hgetall(self._user_key(user_id))
        if not data:
            return None
        
        user = {}
        for k, v in data.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            
            # Try to parse JSON
            try:
                user[key] = json.loads(val)
            except json.JSONDecodeError:
                # Handle datetime fields
                if key in ['created_at', 'last_login']:
                    try:
                        user[key] = datetime.fromisoformat(val)
                    except:
                        user[key] = val
                else:
                    user[key] = val
        
        return user
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username."""
        user_id = self.redis.hget(self.USERNAME_INDEX, username)
        if not user_id:
            return None
        user_id = user_id.decode() if isinstance(user_id, bytes) else user_id
        return self.get_user_by_id(user_id)
    
    def update_user(self, user_id: str, updates: Dict) -> bool:
        """Update user data."""
        if not self.redis.exists(self._user_key(user_id)):
            return False
        
        # Handle username change
        if 'username' in updates:
            old_user = self.get_user_by_id(user_id)
            if old_user:
                self.redis.hdel(self.USERNAME_INDEX, old_user['username'])
                self.redis.hset(self.USERNAME_INDEX, updates['username'], user_id)
        
        self.redis.hset(self._user_key(user_id), mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in updates.items()
        })
        
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        # Remove from username index
        self.redis.hdel(self.USERNAME_INDEX, user['username'])
        
        # Delete user data
        self.redis.delete(self._user_key(user_id))
        
        return True
    
    def list_users(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all users."""
        users = []
        pattern = f"{self.USER_PREFIX}*"
        
        for key in self.redis.scan_iter(match=pattern, count=limit + offset):
            user_id = key.decode().replace(self.USER_PREFIX, "") if isinstance(key, bytes) else key.replace(self.USER_PREFIX, "")
            user = self.get_user_by_id(user_id)
            if user:
                users.append(user)
        
        return users[offset:offset + limit]
    
    # API Key management
    def create_api_key(self, user_id: str, name: str, permissions: List[str], 
                      expires_days: Optional[int] = None) -> Dict:
        """Create a new API key."""
        key_id = secrets.token_hex(8)
        api_key = f"{API_KEY_PREFIX}{secrets.token_hex(32)}"
        
        key_data = {
            'key_id': key_id,
            'name': name,
            'key_hash': hashlib.sha256(api_key.encode()).hexdigest(),
            'user_id': user_id,
            'permissions': json.dumps(permissions),
            'created_at': datetime.utcnow().isoformat(),
            'is_active': 'true'
        }
        
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
            key_data['expires_at'] = expires_at.isoformat()
        
        self.redis.hset(self._api_key(key_id), mapping=key_data)
        
        # Add to user's API keys
        self.redis.sadd(f"{self._user_key(user_id)}:api_keys", key_id)
        
        return {
            'key_id': key_id,
            'name': name,
            'key': api_key,  # Only returned once
            'user_id': user_id,
            'permissions': permissions,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(days=expires_days) if expires_days else None
        }
    
    def get_api_key(self, key_id: str) -> Optional[Dict]:
        """Get API key by ID."""
        data = self.redis.hgetall(self._api_key(key_id))
        if not data:
            return None
        
        key_data = {}
        for k, v in data.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            
            if key == 'permissions':
                key_data[key] = json.loads(val)
            elif key in ['created_at', 'expires_at', 'last_used_at']:
                try:
                    key_data[key] = datetime.fromisoformat(val)
                except:
                    key_data[key] = val
            elif key == 'is_active':
                key_data[key] = val.lower() == 'true'
            else:
                key_data[key] = val
        
        return key_data
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate an API key and return associated user."""
        if not api_key.startswith(API_KEY_PREFIX):
            return None
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Search for matching key hash
        pattern = f"{self.API_KEY_PREFIX}*"
        for key in self.redis.scan_iter(match=pattern):
            key_id = key.decode().replace(self.API_KEY_PREFIX, "") if isinstance(key, bytes) else key.replace(self.API_KEY_PREFIX, "")
            stored_hash = self.redis.hget(self._api_key(key_id), 'key_hash')
            stored_hash = stored_hash.decode() if isinstance(stored_hash, bytes) else stored_hash
            
            if stored_hash == key_hash:
                key_data = self.get_api_key(key_id)
                
                # Check if active and not expired
                if not key_data.get('is_active', False):
                    return None
                
                if key_data.get('expires_at') and key_data['expires_at'] < datetime.utcnow():
                    return None
                
                # Update last used
                self.redis.hset(self._api_key(key_id), 'last_used_at', datetime.utcnow().isoformat())
                
                return key_data
        
        return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if not self.redis.exists(self._api_key(key_id)):
            return False
        
        self.redis.hset(self._api_key(key_id), 'is_active', 'false')
        return True
    
    def list_user_api_keys(self, user_id: str) -> List[Dict]:
        """List all API keys for a user."""
        key_ids = self.redis.smembers(f"{self._user_key(user_id)}:api_keys")
        keys = []
        
        for key_id in key_ids:
            key_id = key_id.decode() if isinstance(key_id, bytes) else key_id
            key_data = self.get_api_key(key_id)
            if key_data:
                # Don't include key hash
                key_data.pop('key_hash', None)
                keys.append(key_data)
        
        return keys

# Initialize user database
user_db = UserDatabase(redis_client)

# =============================================================================
# Authentication Functions
# =============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

def create_refresh_token(data: Dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        user_id = payload.get("sub")
        username = payload.get("username")
        role_str = payload.get("role")
        permissions = payload.get("permissions", [])
        
        if user_id is None:
            return None
        
        role = UserRole(role_str) if role_str else None
        
        return TokenData(
            user_id=user_id,
            username=username,
            role=role,
            permissions=permissions
        )
    except JWTError:
        return None

async def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user by username and password."""
    user = user_db.get_user_by_username(username)
    if not user:
        return None
    
    if not verify_password(password, user.get('hashed_password', '')):
        return None
    
    if not user.get('is_active', True):
        return None
    
    # Update last login
    user_db.update_user(user['id'], {'last_login': datetime.utcnow().isoformat()})
    
    return user

async def get_current_user(token: str) -> Optional[Dict]:
    """Get current user from JWT token."""
    token_data = decode_token(token)
    if not token_data or not token_data.user_id:
        return None
    
    user = user_db.get_user_by_id(token_data.user_id)
    if not user or not user.get('is_active', True):
        return None
    
    return user

async def get_current_user_from_api_key(api_key: str) -> Optional[Dict]:
    """Get current user from API key."""
    key_data = user_db.validate_api_key(api_key)
    if not key_data:
        return None
    
    user = user_db.get_user_by_id(key_data['user_id'])
    return user

def check_permission(user: Dict, permission: Permission) -> bool:
    """Check if user has a specific permission."""
    role = UserRole(user.get('role', UserRole.GUEST))
    
    # Admin has all permissions
    if role == UserRole.ADMIN:
        return True
    
    # Check role permissions
    role_perms = ROLE_PERMISSIONS.get(role, [])
    return permission in role_perms

def require_permission(user: Dict, permission: Permission):
    """Require a permission, raise exception if not present."""
    if not check_permission(user, permission):
        raise PermissionError(f"User does not have permission: {permission}")

# =============================================================================
# Authentication Service
# =============================================================================

class AuthService:
    """Authentication service for user management."""
    
    @staticmethod
    async def register_user(user_create: UserCreate) -> User:
        """Register a new user."""
        # Check if username exists
        existing = user_db.get_user_by_username(user_create.username)
        if existing:
            raise ValueError(f"Username '{user_create.username}' already exists")
        
        # Create user
        hashed_password = get_password_hash(user_create.password)
        
        user_data = {
            'username': user_create.username,
            'email': user_create.email,
            'full_name': user_create.full_name,
            'institution': user_create.institution,
            'hashed_password': hashed_password,
            'role': user_create.role.value,
            'is_active': True
        }
        
        user_id = user_db.create_user(user_data)
        user = user_db.get_user_by_id(user_id)
        
        return User(**user)
    
    @staticmethod
    async def login(login_request: LoginRequest) -> Token:
        """Login user and return tokens."""
        user = await authenticate_user(login_request.username, login_request.password)
        if not user:
            raise ValueError("Invalid username or password")
        
        role = UserRole(user.get('role', UserRole.GUEST))
        permissions = [p.value for p in ROLE_PERMISSIONS.get(role, [])]
        
        token_data = {
            "sub": user['id'],
            "username": user['username'],
            "role": role.value,
            "permissions": permissions
        }
        
        access_token = create_access_token(
            token_data,
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        refresh_token = create_refresh_token(token_data)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    @staticmethod
    async def refresh_access_token(refresh_token: str) -> Token:
        """Refresh access token using refresh token."""
        token_data = decode_token(refresh_token)
        
        if not token_data or not token_data.user_id:
            raise ValueError("Invalid refresh token")
        
        # Verify user still exists and is active
        user = user_db.get_user_by_id(token_data.user_id)
        if not user or not user.get('is_active', True):
            raise ValueError("User not found or inactive")
        
        role = UserRole(user.get('role', UserRole.GUEST))
        permissions = [p.value for p in ROLE_PERMISSIONS.get(role, [])]
        
        new_token_data = {
            "sub": user['id'],
            "username": user['username'],
            "role": role.value,
            "permissions": permissions
        }
        
        access_token = create_access_token(new_token_data)
        new_refresh_token = create_refresh_token(new_token_data)
        
        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    @staticmethod
    async def change_password(user_id: str, old_password: str, new_password: str):
        """Change user password."""
        user = user_db.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        if not verify_password(old_password, user.get('hashed_password', '')):
            raise ValueError("Incorrect current password")
        
        new_hash = get_password_hash(new_password)
        user_db.update_user(user_id, {'hashed_password': new_hash})
    
    @staticmethod
    async def create_api_key(user_id: str, name: str, 
                            permissions: Optional[List[str]] = None,
                            expires_days: Optional[int] = None) -> APIKey:
        """Create API key for user."""
        if not permissions:
            # Use user's role permissions
            user = user_db.get_user_by_id(user_id)
            role = UserRole(user.get('role', UserRole.GUEST))
            permissions = [p.value for p in ROLE_PERMISSIONS.get(role, [])]
        
        key_data = user_db.create_api_key(user_id, name, permissions, expires_days)
        return APIKey(**key_data)
    
    @staticmethod
    async def revoke_api_key(user_id: str, key_id: str) -> bool:
        """Revoke an API key."""
        key_data = user_db.get_api_key(key_id)
        if not key_data or key_data['user_id'] != user_id:
            return False
        
        return user_db.revoke_api_key(key_id)

# =============================================================================
# Decorators
# =============================================================================

def require_auth(permission: Optional[Permission] = None):
    """Decorator to require authentication and optional permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract token from kwargs or args
            token = kwargs.get('token')
            if not token and args:
                for arg in args:
                    if isinstance(arg, str) and len(arg) > 20:
                        token = arg
                        break
            
            if not token:
                raise PermissionError("Authentication required")
            
            user = await get_current_user(token)
            if not user:
                raise PermissionError("Invalid or expired token")
            
            if permission and not check_permission(user, permission):
                raise PermissionError(f"Permission required: {permission}")
            
            # Add user to kwargs
            kwargs['current_user'] = user
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# =============================================================================
# Initialization
# =============================================================================

def init_default_admin():
    """Initialize default admin user if no users exist."""
    users = user_db.list_users(limit=1)
    if not users:
        # Create default admin
        admin_password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin123")
        hashed = get_password_hash(admin_password)
        
        user_data = {
            'username': 'admin',
            'email': 'admin@dftlammps.local',
            'full_name': 'Administrator',
            'hashed_password': hashed,
            'role': UserRole.ADMIN.value,
            'is_active': True
        }
        
        user_id = user_db.create_user(user_data)
        logger.info(f"Created default admin user: admin (ID: {user_id})")
        logger.warning("Please change the default admin password immediately!")

# Initialize on import
init_default_admin()

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Test authentication
    import asyncio
    
    async def test():
        # Create test user
        try:
            user = await AuthService.register_user(UserCreate(
                username="testuser",
                email="test@example.com",
                password="testpass123",
                role=UserRole.RESEARCHER
            ))
            print(f"Created user: {user}")
        except ValueError as e:
            print(f"User creation failed: {e}")
        
        # Login
        try:
            token = await AuthService.login(LoginRequest(
                username="testuser",
                password="testpass123"
            ))
            print(f"Login successful: {token.access_token[:20]}...")
        except ValueError as e:
            print(f"Login failed: {e}")
    
    asyncio.run(test())
