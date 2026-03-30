"""
权限控制中间件
RBAC (Role-Based Access Control) 和 ABAC (Attribute-Based Access Control)
"""

from functools import wraps
from typing import List, Callable, Optional, Any
from fastapi import HTTPException, status, Request
from fastapi.security import SecurityScopes
import logging

logger = logging.getLogger(__name__)


class Permission:
    """权限定义"""
    # 用户管理权限
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # 任务管理权限
    TASK_CREATE = "task:create"
    TASK_READ = "task:read"
    TASK_UPDATE = "task:update"
    TASK_DELETE = "task:delete"
    TASK_CANCEL = "task:cancel"
    
    # DFT计算权限
    DFT_CALCULATE = "dft:calculate"
    DFT_READ = "dft:read"
    DFT_DELETE = "dft:delete"
    
    # MD模拟权限
    MD_SIMULATE = "md:simulate"
    MD_READ = "md:read"
    MD_DELETE = "md:delete"
    
    # ML训练权限
    ML_TRAIN = "ml:train"
    ML_READ = "ml:read"
    ML_DELETE = "ml:delete"
    
    # 筛选权限
    SCREENING_RUN = "screening:run"
    SCREENING_READ = "screening:read"
    
    # 系统管理权限
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"


# 角色权限映射
ROLE_PERMISSIONS = {
    "admin": [
        # 所有权限
        Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE, Permission.USER_DELETE,
        Permission.TASK_CREATE, Permission.TASK_READ, Permission.TASK_UPDATE, Permission.TASK_DELETE, Permission.TASK_CANCEL,
        Permission.DFT_CALCULATE, Permission.DFT_READ, Permission.DFT_DELETE,
        Permission.MD_SIMULATE, Permission.MD_READ, Permission.MD_DELETE,
        Permission.ML_TRAIN, Permission.ML_READ, Permission.ML_DELETE,
        Permission.SCREENING_RUN, Permission.SCREENING_READ,
        Permission.SYSTEM_READ, Permission.SYSTEM_WRITE, Permission.SYSTEM_ADMIN,
    ],
    "researcher": [
        Permission.USER_READ, Permission.USER_UPDATE,
        Permission.TASK_CREATE, Permission.TASK_READ, Permission.TASK_UPDATE, Permission.TASK_CANCEL,
        Permission.DFT_CALCULATE, Permission.DFT_READ,
        Permission.MD_SIMULATE, Permission.MD_READ,
        Permission.ML_TRAIN, Permission.ML_READ,
        Permission.SCREENING_RUN, Permission.SCREENING_READ,
        Permission.SYSTEM_READ,
    ],
    "guest": [
        Permission.TASK_READ,
        Permission.DFT_READ,
        Permission.MD_READ,
        Permission.ML_READ,
        Permission.SCREENING_READ,
    ],
    "api_client": [
        Permission.TASK_CREATE, Permission.TASK_READ, Permission.TASK_CANCEL,
        Permission.DFT_CALCULATE, Permission.DFT_READ,
        Permission.MD_SIMULATE, Permission.MD_READ,
        Permission.ML_TRAIN, Permission.ML_READ,
        Permission.SCREENING_RUN, Permission.SCREENING_READ,
    ],
}


def has_permission(user_role: str, required_permission: str) -> bool:
    """检查角色是否有特定权限"""
    permissions = ROLE_PERMISSIONS.get(user_role, [])
    return required_permission in permissions or Permission.SYSTEM_ADMIN in permissions


def has_any_permission(user_role: str, required_permissions: List[str]) -> bool:
    """检查角色是否有任一权限"""
    return any(has_permission(user_role, perm) for perm in required_permissions)


def has_all_permissions(user_role: str, required_permissions: List[str]) -> bool:
    """检查角色是否有所有权限"""
    return all(has_permission(user_role, perm) for perm in required_permissions)


def check_permission(permission: str):
    """权限检查装饰器"""
    from .security import get_current_user_or_api_key, User
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取当前用户
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not has_permission(current_user.role, permission):
                logger.warning(
                    f"Permission denied: user={current_user.username}, "
                    f"role={current_user.role}, required={permission}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission} required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def check_resource_ownership(get_resource_owner: Callable) -> Callable:
    """资源所有权检查装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from .security import User
            
            current_user = kwargs.get('current_user')
            resource_id = kwargs.get('resource_id')
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # 管理员可以访问所有资源
            if current_user.role == "admin":
                return await func(*args, **kwargs)
            
            # 获取资源所有者
            owner_id = await get_resource_owner(resource_id)
            if owner_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: not the resource owner"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitByUser:
    """基于用户的速率限制"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self._storage = {}  # user_id -> [timestamp, ...]
    
    async def is_allowed(self, user_id: int) -> bool:
        """检查是否允许请求"""
        import time
        from collections import deque
        
        now = time.time()
        window_start = now - 60  # 1分钟窗口
        
        # 获取或创建用户请求历史
        if user_id not in self._storage:
            self._storage[user_id] = deque()
        
        history = self._storage[user_id]
        
        # 移除窗口外的旧请求
        while history and history[0] < window_start:
            history.popleft()
        
        # 检查是否超过限制
        if len(history) >= self.requests_per_minute:
            return False
        
        # 记录当前请求
        history.append(now)
        return True
    
    async def get_remaining(self, user_id: int) -> int:
        """获取剩余请求次数"""
        import time
        from collections import deque
        
        now = time.time()
        window_start = now - 60
        
        if user_id not in self._storage:
            return self.requests_per_minute
        
        history = self._storage[user_id]
        
        # 移除旧请求
        while history and history[0] < window_start:
            history.popleft()
        
        return max(0, self.requests_per_minute - len(history))


# 全局速率限制器实例
user_rate_limiter = RateLimitByUser()


class QuotaManager:
    """用户配额管理器"""
    
    def __init__(self):
        self._usage = {}  # user_id -> daily_usage
        self._last_reset = None
    
    def _check_reset(self):
        """检查是否需要重置每日配额"""
        from datetime import datetime
        
        now = datetime.utcnow()
        if self._last_reset is None or now.date() != self._last_reset.date():
            self._usage = {}
            self._last_reset = now
    
    async def check_quota(self, user_id: int, quota_limit: int, cost: int = 1) -> bool:
        """检查并消耗配额"""
        self._check_reset()
        
        current_usage = self._usage.get(user_id, 0)
        
        if current_usage + cost > quota_limit:
            return False
        
        self._usage[user_id] = current_usage + cost
        return True
    
    async def get_usage(self, user_id: int) -> int:
        """获取当前使用量"""
        self._check_reset()
        return self._usage.get(user_id, 0)
    
    async def get_remaining(self, user_id: int, quota_limit: int) -> int:
        """获取剩余配额"""
        self._check_reset()
        used = self._usage.get(user_id, 0)
        return max(0, quota_limit - used)


# 全局配额管理器
quota_manager = QuotaManager()


class AuditLogger:
    """审计日志记录器"""
    
    @staticmethod
    async def log_action(
        user_id: int,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """记录操作日志"""
        from datetime import datetime
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
        }
        
        # 这里应该写入到持久化存储（数据库、日志文件等）
        logger.info(f"AUDIT: {log_entry}")
        
        # TODO: 写入到审计日志表
        # await database.execute(audit_logs.insert().values(log_entry))


# 全局审计日志记录器
audit_logger = AuditLogger()
