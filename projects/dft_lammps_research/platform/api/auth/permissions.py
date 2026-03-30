"""
Permission and Access Control

Role-based access control (RBAC) and attribute-based access control (ABAC)
for API resources.
"""

from enum import Enum
from typing import List, Dict, Set, Optional
from functools import wraps
from fastapi import HTTPException, status
import structlog

logger = structlog.get_logger()


class Permission(Enum):
    """API Permissions"""
    # Read permissions
    READ_PROJECTS = "projects:read"
    READ_CALCULATIONS = "calculations:read"
    READ_STRUCTURES = "structures:read"
    READ_RESULTS = "results:read"
    
    # Write permissions
    WRITE_PROJECTS = "projects:write"
    WRITE_CALCULATIONS = "calculations:write"
    WRITE_STRUCTURES = "structures:write"
    
    # Admin permissions
    DELETE_RESOURCES = "resources:delete"
    MANAGE_API_KEYS = "apikeys:manage"
    MANAGE_WEBHOOKS = "webhooks:manage"
    VIEW_USAGE = "usage:view"
    
    # Special permissions
    BATCH_OPERATIONS = "batch:execute"
    EXPORT_DATA = "data:export"
    REALTIME_STREAM = "stream:realtime"


class Role(Enum):
    """User Roles"""
    READONLY = "readonly"
    USER = "user"
    DEVELOPER = "developer"
    ADMIN = "admin"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, List[Permission]] = {
    Role.READONLY: [
        Permission.READ_PROJECTS,
        Permission.READ_CALCULATIONS,
        Permission.READ_STRUCTURES,
        Permission.READ_RESULTS,
    ],
    Role.USER: [
        Permission.READ_PROJECTS,
        Permission.READ_CALCULATIONS,
        Permission.READ_STRUCTURES,
        Permission.READ_RESULTS,
        Permission.WRITE_PROJECTS,
        Permission.WRITE_CALCULATIONS,
        Permission.EXPORT_DATA,
    ],
    Role.DEVELOPER: [
        Permission.READ_PROJECTS,
        Permission.READ_CALCULATIONS,
        Permission.READ_STRUCTURES,
        Permission.READ_RESULTS,
        Permission.WRITE_PROJECTS,
        Permission.WRITE_CALCULATIONS,
        Permission.WRITE_STRUCTURES,
        Permission.DELETE_RESOURCES,
        Permission.MANAGE_API_KEYS,
        Permission.MANAGE_WEBHOOKS,
        Permission.BATCH_OPERATIONS,
        Permission.EXPORT_DATA,
        Permission.VIEW_USAGE,
        Permission.REALTIME_STREAM,
    ],
    Role.ADMIN: list(Permission),  # All permissions
}


class PermissionChecker:
    """Check permissions for API access"""
    
    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS
    
    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role"""
        return set(self.role_permissions.get(role, []))
    
    def has_permission(
        self,
        user_permissions: List[str],
        required_permission: Permission
    ) -> bool:
        """Check if user has a specific permission"""
        # Check for wildcard permission
        if "*" in user_permissions or "all" in user_permissions:
            return True
        
        # Check for specific permission
        return required_permission.value in user_permissions
    
    def has_any_permission(
        self,
        user_permissions: List[str],
        required_permissions: List[Permission]
    ) -> bool:
        """Check if user has any of the required permissions"""
        return any(
            self.has_permission(user_permissions, perm)
            for perm in required_permissions
        )
    
    def has_all_permissions(
        self,
        user_permissions: List[str],
        required_permissions: List[Permission]
    ) -> bool:
        """Check if user has all of the required permissions"""
        return all(
            self.has_permission(user_permissions, perm)
            for perm in required_permissions
        )
    
    def require_permission(self, permission: Permission):
        """Decorator to require a specific permission"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract auth context from kwargs
                auth = kwargs.get("auth")
                if not auth:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                user_permissions = auth.get("permissions", [])
                
                if not self.has_permission(user_permissions, permission):
                    logger.warning(
                        "permission_denied",
                        client_id=auth.get("client_id"),
                        required=permission.value,
                        has=user_permissions
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission required: {permission.value}"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def check_resource_access(
        self,
        auth: dict,
        resource_owner: str,
        resource_type: str,
        action: str
    ) -> bool:
        """
        Check if user has access to a specific resource
        
        Implements attribute-based access control (ABAC)
        """
        client_id = auth.get("client_id")
        permissions = auth.get("permissions", [])
        
        # Admin can access everything
        if "*" in permissions or Permission.ADMIN in permissions:
            return True
        
        # Owner can access their own resources
        if client_id == resource_owner:
            return True
        
        # Check specific resource permissions
        resource_perm = f"{resource_type}:{action}"
        if resource_perm in permissions:
            return True
        
        return False
    
    def get_tier_limits(self, tier: str) -> dict:
        """Get API limits for a specific tier"""
        tiers = {
            "free": {
                "rate_limit_per_minute": 60,
                "rate_limit_per_day": 10000,
                "max_projects": 5,
                "max_calculations_per_day": 100,
                "max_structures": 1000,
                "max_webhooks": 1,
                "batch_size": 10,
                "storage_gb": 1,
            },
            "pro": {
                "rate_limit_per_minute": 300,
                "rate_limit_per_day": 100000,
                "max_projects": 50,
                "max_calculations_per_day": 10000,
                "max_structures": 100000,
                "max_webhooks": 10,
                "batch_size": 100,
                "storage_gb": 50,
            },
            "enterprise": {
                "rate_limit_per_minute": 1000,
                "rate_limit_per_day": 1000000,
                "max_projects": -1,  # unlimited
                "max_calculations_per_day": -1,
                "max_structures": -1,
                "max_webhooks": 100,
                "batch_size": 1000,
                "storage_gb": 500,
            },
        }
        return tiers.get(tier, tiers["free"])


# Convenience functions for common permission checks
def require_read(func):
    """Require read permission"""
    checker = PermissionChecker()
    return checker.require_permission(Permission.READ_PROJECTS)(func)


def require_write(func):
    """Require write permission"""
    checker = PermissionChecker()
    return checker.require_permission(Permission.WRITE_PROJECTS)(func)


def require_admin(func):
    """Require admin permission"""
    checker = PermissionChecker()
    return checker.require_permission(Permission.DELETE_RESOURCES)(func)
