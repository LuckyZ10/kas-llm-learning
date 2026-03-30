"""
认证授权模块
"""

from .security import (
    Token,
    TokenData,
    User,
    UserInDB,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_current_active_user,
    get_current_user_or_api_key,
    verify_password,
    get_password_hash,
    generate_api_key,
    refresh_access_token,
    oauth2_scheme,
    api_key_header,
)

from .permissions import (
    Permission,
    ROLE_PERMISSIONS,
    has_permission,
    check_permission,
    check_resource_ownership,
    user_rate_limiter,
    quota_manager,
    audit_logger,
)

__all__ = [
    # 安全
    "Token",
    "TokenData",
    "User",
    "UserInDB",
    "authenticate_user",
    "create_access_token",
    "create_refresh_token",
    "get_current_user",
    "get_current_active_user",
    "get_current_user_or_api_key",
    "verify_password",
    "get_password_hash",
    "generate_api_key",
    "refresh_access_token",
    "oauth2_scheme",
    "api_key_header",
    # 权限
    "Permission",
    "ROLE_PERMISSIONS",
    "has_permission",
    "check_permission",
    "check_resource_ownership",
    "user_rate_limiter",
    "quota_manager",
    "audit_logger",
]
